import numpy as np
import os
from PIL import Image
from matplotlib import image as mpimg
from scipy.io import loadmat

import flatcam
import utils
from multiresolution import *


def preprocess_zero_loc(data_path, std, percent0, percent1, percent2, percent3, SEED0, SEED0N, SEED1, SEED1N, SEED2,
                        SEED2N, SEED3, SEED3N, ):
    calib = loadmat('flatcam_calibdata.mat')
    flatcam.clean_calib(calib)
    root_dir = [x for x in os.walk(os.path.join(data_path, 'noise_added_test_42_sm'))]
#    print(root_dir)
    mulitresolution_zero = os.path.join(data_path, 'zero_multiresolution_42_test_sm_np')
    for sub_dir in root_dir[0][1]:
        noised_dir = os.path.join(mulitresolution_zero, sub_dir)
        if not os.path.isdir(noised_dir):
            os.makedirs(noised_dir)
    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        class_name = os.path.split(class_dir)[1]
        for i, image_name in enumerate(image_list):
            new_image_path = f"{os.path.join(mulitresolution_zero, class_name, image_name.split('.')[0])}.npy"
            if not os.path.exists(new_image_path):
                input_im = mpimg.imread(os.path.join(class_dir, image_name))
                demosaiced_img = flatcam.demosaiced(input_im, calib) * 255
                pil_image = Image.fromarray(demosaiced_img.astype(np.uint8))
                img_128 = pil_image.resize((128, 128))
                dct2_im = dct2Dimg(np.asarray(img_128))
                x0, x1, x2, x3 = divide_dct(dct2_im)

                x0_zn = utils.zero_noise_loc(x0, percent0, SEED0)
                x1_zn = utils.zero_noise_loc(x1, percent1, SEED1)
                x2_zn = utils.zero_noise_loc(x2, percent2, SEED2)
                x3_zn = utils.zero_noise_loc(x3, percent3, SEED3)

                dct2_im_zn = join_quarters(x0_zn, x1_zn, x2_zn, x3_zn)
                dct2_zn_im = Image.fromarray(dct2_im_zn.astype(np.uint8))
#                dct2_zn_im.save('dct_128.png')
                idct2_im_zn = idct2Dimg(dct2_im_zn)
                idct_im_pil_zn = Image.fromarray(idct2_im_zn.astype(np.uint8))
                idct_im_64 = idct_im_pil_zn.resize((64, 64))
                dct_64 = dct2Dimg(np.asarray(idct_im_64))
                # dct_64_pil = Image.fromarray(dct_64.astype(np.uint8))
                # dct_64_pil.save("dct_zn_64.png")
                multiresolution_data = np.concatenate((x0_zn.astype(np.uint8), x1_zn.astype(np.uint8),
                                                       x2_zn.astype(np.uint8), x3_zn.astype(np.uint8),
                                                       dct_64.astype(np.uint8)), axis=2)
                np.save(os.path.join(mulitresolution_zero, class_name, image_name.split('.')[0]), multiresolution_data)
        print(f"Added Noise in class: {class_dir.split('/')[-1]}")



def preprocess_zero_loc_path(data_path, loc0, loc1, loc2, loc3):
    calib = loadmat('flatcam_calibdata.mat')
    flatcam.clean_calib(calib)
    root_dir = [x for x in os.walk(os.path.join(data_path, 'test_dataset_42_sm'))]
    mulitresolution_zero = os.path.join(data_path, 'zero_multiresolution_42_test_sigma_2per')
    for sub_dir in root_dir[0][1]:
        noised_dir = os.path.join(mulitresolution_zero, sub_dir)
        if not os.path.isdir(noised_dir):
            os.makedirs(noised_dir)
    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        class_name = os.path.split(class_dir)[1]
        for i, image_name in enumerate(image_list):
            new_image_path = f"{os.path.join(mulitresolution_zero, class_name, image_name.split('.')[0])}.npy"
            if not os.path.exists(new_image_path):
                input_im = mpimg.imread(os.path.join(class_dir, image_name))
                demosaiced_img = flatcam.demosaiced(input_im, calib) * 255
                pil_image = Image.fromarray(demosaiced_img.astype(np.uint8))
                img_128 = pil_image.resize((128, 128))
                dct2_im = dct2Dimg(np.asarray(img_128))
                x0, x1, x2, x3 = divide_dct(dct2_im)

                x0_zn = utils.zero_noise_loc_path(x0, loc0)
                x1_zn = utils.zero_noise_loc_path(x1, loc1)
                x2_zn = utils.zero_noise_loc_path(x2, loc2)
                x3_zn = utils.zero_noise_loc_path(x3, loc3)

                dct2_im_zn = join_quarters(x0_zn, x1_zn, x2_zn, x3_zn)
                idct2_im_zn = idct2Dimg(dct2_im_zn)
                idct_im_pil_zn = Image.fromarray(idct2_im_zn.astype(np.uint8))
                idct_im_64 = idct_im_pil_zn.resize((64, 64))
                dct_64 = dct2Dimg(np.asarray(idct_im_64))
                multiresolution_data = np.concatenate((x0_zn.astype(np.uint8), x1_zn.astype(np.uint8),
                                                       x2_zn.astype(np.uint8), x3_zn.astype(np.uint8),
                                                       dct_64.astype(np.uint8)), axis=2)
                np.save(os.path.join(mulitresolution_zero, class_name, image_name.split('.')[0]), multiresolution_data)
        print(f"Added Noise in class: {class_dir.split('/')[-1]}")


def preprocess_zero_loc_path_new(data_path, k , per):
    calib = loadmat('flatcam_calibdata.mat')
    flatcam.clean_calib(calib)
    root_dir = [x for x in os.walk(os.path.join(data_path, 'test_dataset_42_sm'))]
    mulitresolution_zero = os.path.join(data_path, f'new_zero_multiresolution_42_idct_{int(k*10)}_{per}')
    mulitresolution_noise = os.path.join(data_path, f'new_noise_multiresolution_42_{int(k*10)}_{per}')
    image_noise = os.path.join(data_path, f'new_noise_image_42_{int(k*10)}_{per}')
    for sub_dir in root_dir[0][1]:
       noised_dir = os.path.join(image_noise, sub_dir)
       if not os.path.isdir(noised_dir):
           os.makedirs(noised_dir)
    for sub_dir in root_dir[0][1]:
        noised_dir = os.path.join(mulitresolution_zero, sub_dir)
        if not os.path.isdir(noised_dir):
            os.makedirs(noised_dir)
    for sub_dir in root_dir[0][1]:
       noised_dir = os.path.join(mulitresolution_noise, sub_dir)
       if not os.path.isdir(noised_dir):
           os.makedirs(noised_dir)
    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        class_name = os.path.split(class_dir)[1]
        for i, image_name in enumerate(image_list):
            new_image_path = f"{os.path.join(mulitresolution_zero, class_name, image_name.split('.')[0])}.npy"
            if not os.path.exists(new_image_path):
                new_noise_path = os.path.join(mulitresolution_noise, class_name, image_name.split('.')[0])
                new_multi_path = os.path.join(mulitresolution_zero, class_name, image_name.split('.')[0])
                image_path = os.path.join(image_noise, class_name, image_name.split('.')[0])
                path = os.path.join(class_dir, image_name)
                utils.add_noise(path, new_noise_path, new_multi_path, image_path+'.png', k, per)
        print(f"Added Noise in class: {class_dir.split('/')[-1]}")


def tests(data_path, std, percent0, percent1, percent2, percent3, SEED0, SEED0N, SEED1, SEED1N, SEED2,
                        SEED2N, SEED3, SEED3N, ):
    calib = loadmat('flatcam_calibdata.mat')
    flatcam.clean_calib(calib)
    input_im = mpimg.imread(data_path)
    demosaiced_img = flatcam.demosaiced(input_im, calib) * 255
    pil_image = Image.fromarray(demosaiced_img.astype(np.uint8))
    # pil_image = Image.open('demos_noise.png')
    img_128 = pil_image.resize((128, 128))
    dct2_im = dct2Dimg(np.asarray(img_128))
    # dct2_ims = dct2_im.astype(np.uint8)
    x0, x1, x2, x3 = divide_dct(dct2_im)
    x0_n = utils.add_noise_percent(x0, std, percent0, SEED0, SEED0N)
    x1_n = utils.add_noise_percent(x1, std, percent1, SEED1, SEED1N)
    x2_n = utils.add_noise_percent(x2, std, percent2, SEED2, SEED2N)
    x3_n = utils.add_noise_percent(x3, std, percent3, SEED3, SEED3N)

    x0_zn = utils.zero_noise_loc(x0, percent0, SEED0)
    x1_zn = utils.zero_noise_loc(x1, percent1, SEED1)
    x2_zn = utils.zero_noise_loc(x2, percent2, SEED2)
    x3_zn = utils.zero_noise_loc(x3, percent3, SEED3)

    dct2_im_n = join_quarters(x0_n, x1_n, x2_n, x3_n)
    idct2_im_n = idct2Dimg(dct2_im_n)
    idct_im_pil = Image.fromarray(idct2_im_n.astype(np.uint8))
    idct_im_pil = idct_im_pil.resize((620, 500))
    idct_im_pil.save('demos_noise.png')
    sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil), calib) * 255
    lmbd = 3e-4  # L2 regularization parameter
    recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    recon_images = Image.fromarray(recon_image.astype(np.uint8))
    recon_images.save('noise_recon.png')

    dct2_im_zn = join_quarters(x0_zn, x1_zn, x2_zn, x3_zn)
    dct2_im_zns = dct2_im_zn
    dct2_zn_im = Image.fromarray(dct2_im_zns.astype(np.uint8))
    dct2_zn_im.save('dct_zn.png')
    idct2_im_zn = idct2Dimg(dct2_im_zn)
    idct_im_pil_zn = Image.fromarray(idct2_im_zn.astype(np.uint8))
    idct_im_64 = idct_im_pil_zn.resize((64, 64))
    dct_64 = dct2Dimg(np.asarray(idct_im_64))
    dct_64 = Image.fromarray(dct_64.astype(np.uint8))
    dct_64.save("dct_zn_64.png")
    idct_im_pil_zn = idct_im_pil_zn.resize((620, 500))
    idct_im_pil_zn.save('demos_zn.png')
    sensor_measurement_zn = flatcam.demosaic_fc(np.asarray(idct_im_pil_zn), calib) * 255
    lmbd = 3e-4  # L2 regularization parameter
    recon_image_zn = flatcam.fcrecon(sensor_measurement_zn, calib, lmbd) * 255
    recon_images_zn = Image.fromarray(recon_image_zn.astype(np.uint8))
    recon_images_zn.save('zn_recon.png')


if __name__ == '__main__':
    data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face'
    loc0 = 'variances/Y0_loc.npy'
    loc1 = 'variances/Y1_loc.npy'
    loc2 = 'variances/Y2_loc.npy'
    loc3 = 'variances/Y3_loc.npy'
    ks = [0.3, 0.5, 0.7, 0.9]
    pers = [5]
    for k in ks:
        for per in pers:
            preprocess_zero_loc_path_new(data_path, k, per)


    # preprocess_zero_loc(data_path, 0.2, 1, 5, 5, 10, 13, 25, 18, 29, 67, 42, 88, 108)

    # data_path = 'orig/001.png'
#    data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face/noise_added_test_42_sm/01/002.png'
#     tests(data_path, 0.2, 1, 5, 5, 10, 13, 25, 18, 29, 67, 42, 88, 108)
