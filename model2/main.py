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
from resnet import *
from utils import progress_bar, get_mean_and_std
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--mode', type=str, default='pc', help='attention mode')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def npy_loader(path):
    sample = np.load(path)
    return sample


class Numpy_Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path='../NWPU-RESISC45/', transform=None):
        self.dataset = torchvision.datasets.DatasetFolder(
            root=data_path,
            loader=npy_loader,
            extensions='.npy'
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


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

data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face/demosaiced_measurement_np'
data_path_train = '/scratch/s571b087/project/Lensless_Imaging/rice_face/flatcam_split_dataset_np/train'
data_path_test = '/scratch/s571b087/project/Lensless_Imaging/rice_face/flatcam_split_dataset_np/test'
# data_path = '/home/s571b087/lensless/project/rice_face/demosaiced_measurement'
# data_path_train = '/home/s571b087/lensless/project/rice_face/flatcam_split_dataset/train'
# data_path_test = '/home/s571b087/lensless/project/rice_face/flatcam_split_dataset/test'

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

mean = (0.22008048, 0.17645225, 0.17293943, 0.07200571, 0.046010967, 0.041765355, 0.06722115, 0.039112534, 0.036752794, 0.019486148, 0.0096396115, 0.0100624915, 0.11260818, 0.08309171, 0.078678414)
std = (0.40342093, 0.37246585, 0.3694478, 0.2515716, 0.20276384, 0.19349617, 0.24132505, 0.1852777, 0.17997105, 0.13187744, 0.0916155, 0.09458283, 0.30751574, 0.26933154, 0.26219055)
transform_train_np = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test_np = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_data = Numpy_Dataset(data_path=data_path_train, transform=transform_train_np)
test_data = Numpy_Dataset(data_path=data_path_test, transform=transform_test_np)

# train_data = NWPUDataset(data_path=data_path_train, transform=transform_train)
# test_data = NWPUDataset(data_path=data_path_test, transform=transform_test)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          sampler=None)
testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                         sampler=None)
lr_str = str(args.lr)
log_path = f'np_{args.outf}_lr_{lr_str.split(".")[-1]}_bs_{args.batch_size}_attn_mode_{args.mode}'
writer = SummaryWriter(log_path)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
elif args.mode == 'pc' or args.mode == 'dp':
    print('==> Building model..')
    net = VGG_ATT(mode=args.mode)
elif args.mode.startswith('vgg'):
    vgg_cnf = args.mode.upper()
    print('==> Building VGG model..')
    net = VGG(vgg_cnf)
else:
    print('==> Building Resnet model..')
    net = ResNet18()

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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    writer.add_scalar('train/loss', round(train_loss / (batch_idx + 1), 2), epoch)
    writer.add_scalar('train/accuracy', 100. * correct / total, epoch)
    scheduler.step()


@torch.no_grad()
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    writer.add_scalar('test/accuracy', 100. * correct / total, epoch)
    writer.add_scalar('test/loss', round((test_loss / (batch_idx + 1)), 2), epoch)

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
