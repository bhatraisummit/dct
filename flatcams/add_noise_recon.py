import os

from matplotlib import image as mpimg
from scipy.io import loadmat
from PIL import Image
import numpy as np
import utils
from multiresolution import *

import flatcam


def preprocess_add_noise(data_path, std, percent0, percent1, percent2, percent3, SEED0, SEED0N, SEED1, SEED1N, SEED2, SEED2N, SEED3, SEED3N, ):
    root_dir = [x for x in os.walk(os.path.join(data_path, 'noise_test'))]
    root_new = os.path.join(data_path, 'noise_added_test')
    root_recons = os.path.join(data_path, 'noise_recon')
    for sub_dir in root_dir[0][1]:
        recon_dir = os.path.join(root_recons, sub_dir)
        if not os.path.isdir(recon_dir):
            os.makedirs(recon_dir)
    for sub_dir in root_dir[0][1]:
        noised_dir = os.path.join(root_new, sub_dir)
        if not os.path.isdir(noised_dir):
            os.makedirs(noised_dir)
    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        class_name = os.path.split(class_dir)[1]
        for i, image_name in enumerate(image_list):
            new_image_path = f"{os.path.join(root_new, class_name, image_name.split('.')[0])}.png"
            if not os.path.exists(new_image_path):
                input_im = mpimg.imread(os.path.join(class_dir, image_name))
                calib = loadmat('flatcam_calibdata.mat')
                demosaiced_img = flatcam.demosaiced(input_im, calib) * 255
                pil_image = Image.fromarray(demosaiced_img.astype(np.uint8))
                img_128 = pil_image.resize((128, 128))
                dct2_im = dct2Dimg(np.asarray(img_128) / 255)
                x0, x1, x2, x3 = divide_dct(dct2_im)
                x0_n = utils.add_noise_percent(x0, std, percent0, SEED0, SEED0N)
                x1_n = utils.add_noise_percent(x1, std, percent1, SEED1, SEED1N)
                x2_n = utils.add_noise_percent(x2, std, percent2, SEED2, SEED2N)
                x3_n = utils.add_noise_percent(x3, std, percent3, SEED3, SEED3N)
                dct2_im_n = join_quarters(x0_n, x1_n, x2_n, x3_n)
                idct2_im_n = idct2Dimg(dct2_im_n) * 255
                idct_im_pil = Image.fromarray(idct2_im_n.astype(np.uint8))
                new_path = os.path.join(root_new, class_name, image_name)
                idct_im_pil.save(new_path)
                idct_im_pil = idct_im_pil.resize((620, 500))
                sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil), calib) * 255
                lmbd = 3e-4  # L2 regularization parameter
                recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
                recon_images = Image.fromarray(recon_image.astype(np.uint8))
                recon_path = os.path.join(root_recons, class_name, image_name)
                recon_images.save(recon_path)
            # imageio.imwrite(os.path.join(root_new, class_name, image_name), demosaiced_img.astype(np.uint8))
        print(f"Added Noise in class: {class_dir.split('/')[-1]}")


def tests():
    img_path = '001.png'
    img = mpimg.imread(img_path)
    calib = loadmat('flatcam_calibdata.mat')
    dem_img = flatcam.demosaiced(img, calib) * 255
    pil_image = Image.fromarray(dem_img.astype(np.uint8))
    pil_image.save('pil.png')


if __name__ == '__main__':
    data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face'
    preprocess_add_noise(data_path, 0.2, 1, 5, 5, 10, 13, 25, 18, 29, 67, 42, 88, 108)

