import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import SubsetRandomSampler

from multiresolutionTransform import MultiResolution

import os
import argparse
import numpy as np

from vgg import *
from utils import progress_bar, get_mean_and_std
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


class NWPUDataset(torch.utils.data.Dataset):

    def __init__(self, data_path='../NWPU-RESISC45/', transform=None):
        self.dataset = torchvision.datasets.ImageFolder(root=data_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Data
print('==> Preparing data..')

data_path = '/home/s571b087/lensless/project/rice_face/demosaiced_measurement'
data_path_train = '/home/s571b087/lensless/project/rice_face/flatcam_split_dataset/train'
data_path_test = '/home/s571b087/lensless/project/rice_face/flatcam_split_dataset/test'

# transform_data = transforms.Compose([
#     transforms.ToTensor()
# ])
# dataset = NWPUDataset(data_path=data_path, transform=transform_data)
# mean, std = get_mean_and_std(dataset)
# print(f'Dataset Mean: {mean}, std: {std}')
im_size = 32
num_classes = 87
num_images_per_class = 274

transform_train = transforms.Compose([
    MultiResolution(),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    MultiResolution(),
    transforms.ToTensor()
])

train_data = NWPUDataset(data_path=data_path_train, transform=transform_train)
test_data = NWPUDataset(data_path=data_path_test, transform=transform_test)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          sampler=None)
testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                         sampler=None)

writer = SummaryWriter(args.outf)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = VGG_ATT(mode='pc')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, 50)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    accuracy = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        accuracy = 100. * correct / total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/accuracy', accuracy, epoch)
    scheduler.step()


@torch.no_grad()
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc = 0
    loss = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss = test_loss / (batch_idx + 1)
        acc = 100. * correct / total
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    writer.add_scalar('test/accuracy', acc, epoch)
    writer.add_scalar('test/loss', loss, epoch)

    # Save checkpoint.
    acc = 100. * correct / total

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
