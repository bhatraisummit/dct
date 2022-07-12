from PIL import Image
from skimage import metrics
import numpy as np
import os

def psnr (orig_path, modi_path):
    orig_im = Image.open(orig_path)
    orig_im = np.asarray(orig_im)
    modi_im = Image.open(modi_path)
    modi_im = np.asarray(modi_im)
    return metrics.peak_signal_noise_ratio(orig_im, modi_im)


def ssim (orig_path, modi_path):
    orig_im = Image.open(orig_path)
    orig_im = np.asarray(orig_im)
    modi_im = Image.open(modi_path)
    modi_im = np.asarray(modi_im)
    return metrics.structural_similarity(orig_im, modi_im, multichannel=True)

def image_diff (orig_paths, modi_paths):
    orig_dir = [x for x in os.walk(orig_paths)]
    for c, sub_dir in enumerate(orig_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        class_name = os.path.split(class_dir)[1]
        psnrs = []
        ssims = []
        for i, image_name in enumerate(image_list):
            orig_path = os.path.join(class_dir, image_name)
            modi_path = os.path.join(modi_paths, class_name, image_name)
            psnrs.append(psnr(orig_path, modi_path))
            ssims.append(ssim(orig_path, modi_path))
    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)

def separate_test_reconstructed(data_path):
    root_dir = [x for x in os.walk(os.path.join(data_path, 'reconstruct_rice'))]
    root_new = os.path.join(data_path, 'reconstruct_orig_test')
    test_path = '../model2/test_path.txt'
    test_paths = []
    with open(test_path, 'r') as fout:
        lines = fout.readlines()
        for line in lines:
            test_paths.append(line.strip())
#    print(test_paths[0])
    for sub_dir in root_dir[0][1]:
        test_dir = os.path.join(root_new, sub_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        class_name = os.path.split(class_dir)[1]
        split_paths = class_dir.split('reconstruct_rice')
        for i, image_name in enumerate(image_list):
            npy_path = image_name.split('.')[0]+'.npy'
            path_expected_in_file = f'{split_paths[0]}demosaiced_measurement_np_64{split_paths[-1]}/{npy_path}'
#            print(path_expected_in_file)
            source_im = os.path.join(class_dir, image_name)
            if path_expected_in_file in test_paths:
                dest_im = os.path.join(root_new, class_name, image_name)
                os.system(f'cp {source_im} {dest_im}')
        print(f"Done for {class_name}")

#data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face'
#separate_test_reconstructed(data_path)

ks = [0.3, 0.5, 0.7, 0.9]
pers = [2,3,4,5]
for k in ks:
    for per in pers:
        recon_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face/reconstruct_orig_test'
        modi_path = f'/scratch/s571b087/project/Lensless_Imaging/rice_face/new_noise_image_42_{int(k*10)}_{per}'
        avg_psnr, avg_ssim = image_diff(recon_path, modi_path)
        print(f'for std {k} and percent {per} avg psnr= {avg_psnr} and avg ssim = {avg_ssim}')
