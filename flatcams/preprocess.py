import os

from matplotlib import image as mpimg
from scipy.io import loadmat
import numpy as np
import cv2

import flatcam
import imageio


def preprocess_demosaic(data_path):
    root_dir = [x for x in os.walk(os.path.join(data_path, 'tests'))]
    root_new = os.path.join(data_path, 'demosaiced_measurement')
    for sub_dir in root_dir[0][1]:
        demosaic_dir = os.path.join(root_new, sub_dir)
        if not os.path.isdir(demosaic_dir):
            os.makedirs(demosaic_dir)
    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        for i, image_name in enumerate(image_list):
            input_im = mpimg.imread(os.path.join(class_dir, image_name))
            calib = loadmat('flatcam_calibdata.mat')
            demosaiced_img = flatcam.demosaiced(input_im, calib)
            demosaiced_img *= 255
            demosaiced_img = cv2.resize(demosaiced_img, (64, 64), interpolation=cv2.INTER_CUBIC)
            class_name = os.path.split(class_dir)[1]
            imageio.imwrite(os.path.join(root_new, class_name, image_name), demosaiced_img.astype(np.uint8))
            print(imageio.imread(os.path.join(root_new, class_name, image_name)).shape)
        print(f"Conversion done for {class_dir.split('/')[-1]}")


def tests():
    img_path = '001.png'
    img = mpimg.imread(img_path)
    calib = loadmat('flatcam_calibdata.mat')
    dem_img = flatcam.demosaiced(img, calib)
    mpimg.imsave('demosaiced_plt.jpg', dem_img)
    dem = mpimg.imread('demosaiced_plt.jpg')
    fc = flatcam.demosaic_fc(dem, calib)
    mpimg.imsave('fc.png', fc)
    sen_img = mpimg.imread('fc.png')
    recon = flatcam.fcrecon(sen_img, calib, 3e-4)
    mpimg.imsave('recon.jpg', recon)


if __name__ == '__main__':
    preprocess_demosaic('/Users/summit/Desktop')
    # preprocess_demosaic('/scratch/s571b087/project/Lensless_Imaging/rice_face')
    # preprocess_demosaic('/home/s571b087/lensless/project/rice_face')
    # tests()
