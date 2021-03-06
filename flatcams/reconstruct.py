import os

from matplotlib import image as mpimg
from scipy.io import loadmat
from PIL import Image
import numpy as np
from multiresolution import multiresolution_dct

import flatcam


def preprocess_reconstruct(data_path):
    root_dir = [x for x in os.walk(os.path.join(data_path, 'fc_captures'))]
    root_new = os.path.join(data_path, 'reconstruct_rice')
    calib = loadmat('flatcam_calibdata.mat')
    flatcam.clean_calib(calib)
    lmbd = 3e-4
    for sub_dir in root_dir[0][1]:
        demosaic_dir = os.path.join(root_new, sub_dir)
        if not os.path.isdir(demosaic_dir):
            os.makedirs(demosaic_dir)
    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        class_name = os.path.split(class_dir)[1]
        for i, image_name in enumerate(image_list):
            new_image_path = f"{os.path.join(root_new, class_name, image_name.split('.')[0])}.png"
            if not os.path.exists(new_image_path):
                input_im = mpimg.imread(os.path.join(class_dir, image_name))
                recon_image = flatcam.fcrecon(input_im, calib, lmbd) * 255
                pil_image = Image.fromarray(recon_image.astype(np.uint8))
                pil_image.save(new_image_path)
            # imageio.imwrite(os.path.join(root_new, class_name, image_name), demosaiced_img.astype(np.uint8))
        print(f"Conversion done for {class_dir.split('/')[-1]}")


def tests():
    img_path = '001.png'
    img = mpimg.imread(img_path)
    calib = loadmat('flatcam_calibdata.mat')
    dem_img = flatcam.demosaiced(img, calib) * 255
    pil_image = Image.fromarray(dem_img.astype(np.uint8))
    pil_image.save('pil.png')

if __name__ == '__main__':
    preprocess_reconstruct('/scratch/s571b087/project/Lensless_Imaging/rice_face')
