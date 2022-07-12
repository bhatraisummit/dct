import numpy as np
import random
import math
import flatcam
from scipy.io import loadmat
import matplotlib.image as mpimg
from PIL import Image
import multiresolution
import time

def add_noise_percent(arr, std, percent, SEED, SEED_N):
    np.random.seed(SEED_N)
    noise = np.random.normal(0, std, size=arr.shape)
    flat_noise = noise.flatten()
    random.seed(SEED)
    indexs = random.sample(range(flat_noise.shape[0]), math.ceil((100 - percent) / 100. * flat_noise.shape[0]))
    flat_noise[indexs] = 0
    return arr + flat_noise.reshape(arr.shape)

def zero_noise_loc (arr, percent, SEED):
    final_shape = arr.shape
    total_size = 1
    for i in final_shape:
        total_size *= i
    random.seed(SEED)
    indexs = random.sample(range(total_size), math.ceil((100 - percent) / 100. * total_size))
    locs = np.where(~np.in1d(range(total_size),indexs))
    arr = arr.flatten()
    arr[locs] = 0
    return arr.reshape(final_shape)

def zero_noise_loc_path (arr, path):
    final_shape = arr.shape
    total_size = 1
    for i in final_shape:
        total_size *= i
    locs = np.load(path)
    arr = arr.flatten()
    arr[locs] = 0
    return arr.reshape(final_shape)

def remove_noise_percent (arr, std, percent, SEED, SEED_N):
    np.random.seed(SEED_N)
    noise = np.random.normal(0, std, size=arr.shape)
    flat_noise = noise.flatten()
    random.seed(SEED)
    indexs = random.sample(range(flat_noise.shape[0]), math.ceil((100 - percent) / 100. * flat_noise.shape[0]))
    flat_noise[indexs] = 0
    arr = arr - flat_noise.reshape(arr.shape)
    return arr


def add_noise_std (arr, loc, std, SEED):
    t0 = time.time()
    np.random.seed(SEED)
    noise = np.random.normal(0, std, size=arr.shape)
    flat_noise = noise.flatten()
    reshape_size = arr.shape
    arr = arr.flatten()
    arr[loc] = arr[loc] + flat_noise [loc]
    arr1 = arr.reshape(reshape_size)
    t1 = time.time()
    zero_arr = arr.copy()
    zero_arr[loc] = 0
    return arr1, zero_arr.reshape(reshape_size), t1-t0


def zero_loc(path, new_multi_path,  k, per):
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    flatcam.clean_calib(calib)
    Y0_path = f'variances/Y0_loc_{per}.npy'
    Y1_path = f'variances/Y1_loc_{per}.npy'
    Y2_path = f'variances/Y2_loc_{per}.npy'
    Y3_path = f'variances/Y3_loc_{per}.npy'


    Y0_loc = np.load(Y0_path)
    Y1_loc = np.load(Y1_path)
    Y2_loc = np.load(Y2_path)
    Y3_loc = np.load(Y3_path)

    input_im = mpimg.imread(path).astype(float)
    demosaiced_im = flatcam.demosaiced(input_im, calib) * 255
    pil_image = Image.fromarray(demosaiced_im.astype(np.uint8))
    img_128 = pil_image.resize((128, 128))
    dct2_im = multiresolution.dct2Dimg(np.asarray(img_128)/255)
    x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)

    std = k
    x0, x0n, t0 = add_noise_std(x0, Y0_loc, std, 13)
    x1, x1n, t1 = add_noise_std(x1, Y1_loc, std, 25)
    x2, x2n, t2 = add_noise_std(x2, Y2_loc, std, 18)
    x3, x3n, t3 = add_noise_std(x3, Y3_loc, std, 29)
    dct2_im = multiresolution.join_quarters(x0, x1, x2, x3)
    idct2_im_n = multiresolution.idct2Dimg(dct2_im) * 255
    idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    dct2_im = multiresolution.dct2Dimg(np.asarray(idct_im_pil_n)/255)
    x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)
    x0 = zero_noise_loc_path (x0, Y0_path)
    x1 = zero_noise_loc_path (x1, Y1_path)
    x2 = zero_noise_loc_path (x2, Y2_path)
    x3 = zero_noise_loc_path (x3, Y3_path)
    dct2_im = multiresolution.join_quarters(x0, x1, x2, x3)
    idct2_im_n = multiresolution.idct2Dimg(dct2_im) * 255
    idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    multiresolution_data = multiresolution.multiresolution_dct(idct_im_pil_n)
    np.save(new_multi_path, multiresolution_data)



def add_noise(path, new_noise_path, new_multi_path, image_path, k, per):
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    flatcam.clean_calib(calib)
    Y0_path = f'variances/Y0_loc_{per}.npy'
    Y1_path = f'variances/Y1_loc_{per}.npy'
    Y2_path = f'variances/Y2_loc_{per}.npy'
    Y3_path = f'variances/Y3_loc_{per}.npy'


    Y0_loc = np.load(Y0_path)
    Y1_loc = np.load(Y1_path)
    Y2_loc = np.load(Y2_path)
    Y3_loc = np.load(Y3_path)

    input_im = mpimg.imread(path).astype(float)
    demosaiced_im = flatcam.demosaiced(input_im, calib) * 255
    pil_image = Image.fromarray(demosaiced_im.astype(np.uint8))
    img_128 = pil_image.resize((128, 128))
    dct2_im = multiresolution.dct2Dimg(np.asarray(img_128)/255)
    x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)

    std = k
    x0, x0n, t0 = add_noise_std(x0, Y0_loc, std, 13)
    x1, x1n, t1 = add_noise_std(x1, Y1_loc, std, 25)
    x2, x2n, t2 = add_noise_std(x2, Y2_loc, std, 18)
    x3, x3n, t3 = add_noise_std(x3, Y3_loc, std, 29)
    dct2_im = multiresolution.join_quarters(x0, x1, x2, x3)
    idct2_im_n = multiresolution.idct2Dimg(dct2_im) * 255
    idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    multiresolution_data = multiresolution.multiresolution_dct(idct_im_pil_n)
    np.save(new_noise_path, multiresolution_data)

    idct_im_pil_ns = idct_im_pil_n.resize((620, 500))
    sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil_ns), calib) * 255
    lmbd = 3e-4  # L2 regularization parameter
    recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    recon_images_n = Image.fromarray(recon_image.astype(np.uint8))
    recon_images_n.save(image_path)

    dct2_im = multiresolution.dct2Dimg(np.asarray(idct_im_pil_n) / 255)
    x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)
    x0 = zero_noise_loc_path(x0, Y0_path)
    x1 = zero_noise_loc_path(x1, Y1_path)
    x2 = zero_noise_loc_path(x2, Y2_path)
    x3 = zero_noise_loc_path(x3, Y3_path)
    dct2_im = multiresolution.join_quarters(x0, x1, x2, x3)
    idct2_im_n = multiresolution.idct2Dimg(dct2_im) * 255
    idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    multiresolution_data = multiresolution.multiresolution_dct(idct_im_pil_n)
    np.save(new_multi_path, multiresolution_data)







