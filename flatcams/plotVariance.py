import cv2
import imageio
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from PIL import Image
from scipy.io import loadmat
from skimage import metrics

import flatcam
import multiresolution
import utils


def variance_quarters(data_path):
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    flatcam.clean_calib(calib)
    root_dir = [x for x in os.walk(os.path.join(data_path, 'rice_rand_87'))]
    X0 = []
    X1 = []
    X2 = []
    X3 = []
    class_dir = root_dir[0][0]
    image_list = root_dir[0][2]
    for i, image_name in enumerate(image_list):
        path = os.path.join(class_dir, image_name)
        input_im = mpimg.imread(path).astype(float)
        demosaiced_im = flatcam.demosaiced(input_im, calib) * 255
        pil_image = Image.fromarray(demosaiced_im.astype(np.uint8))
        img_128 = pil_image.resize((128, 128))
        dct2_im = multiresolution.dct2Dimg(np.asarray(img_128))
        x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)
        X0.append(x0.flatten())
        X1.append(x1.flatten())
        X2.append(x2.flatten())
        X3.append(x3.flatten())
    X0 = np.asarray(X0)
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    X3 = np.asarray(X3)
    range_len = 64 * 64 * 3
    Y0 = []
    Y1 = []
    Y2 = []
    Y3 = []
    for i in range(range_len):
        Y0.append(np.var(X0[:, i], dtype=np.float64))
        Y1.append(np.var(X1[:, i], dtype=np.float64))
        Y2.append(np.var(X2[:, i], dtype=np.float64))
        Y3.append(np.var(X3[:, i], dtype=np.float64))
    np.save('variances/Y0', Y0)
    np.save('variances/Y1', Y1)
    np.save('variances/Y2', Y2)
    np.save('variances/Y3', Y3)


def save_hist(path):
    var = np.load(path).astype(np.float64)
    fig = plt.figure()
    plot = plt.hist(var, density=False, bins=30)
    print(path, plot[0])
    print(path, plot[1])
    fig.savefig(f'{path.split(".")[0]}.png')
    plt.close(fig)


def find_locations_max(path, percent):
    var = np.load(path).astype(np.float64)
    no_of_locs = math.floor(percent * var.shape[0] / 100)
    locs = np.argpartition(var, -no_of_locs)[-no_of_locs:]
    print('var', var[locs])
    return locs

def find_locations_max_no(arr, no_of_locs):
    locs = np.argpartition(arr, -no_of_locs)[-no_of_locs:]
    return locs

def find_locations_min(path, percent):
    var = np.load(path).astype(np.float64)
    no_of_locs = math.floor(percent * var.shape[0] / 100)
    return np.argpartition(var, no_of_locs)[:no_of_locs]


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

def remove_noise_std (arr, loc, std, SEED):
    t0 = time.time()
    np.random.seed(SEED)
    noise = np.random.normal(0, std, size=arr.shape)
    flat_noise = noise.flatten()
    reshape_size = arr.shape
    arr = arr.flatten()
    arr[loc] = arr[loc] - flat_noise [loc]
    arr1 = arr.reshape(reshape_size)
    t1 = time.time()
    return arr1, t1-t0


def add_noise(path, k, per):
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    flatcam.clean_calib(calib)
    Y0_path = f'variances/Y0_loc_{per}.npy'
    Y1_path = f'variances/Y0_loc_{per}.npy'
    Y2_path = f'variances/Y0_loc_{per}.npy'
    Y3_path = f'variances/Y0_loc_{per}.npy'


    # Y0_path = 'variance/Y0_loc_min.npy'
    # Y1_path = 'variance/Y1_loc_min.npy'
    # Y2_path = 'variance/Y2_loc_min.npy'
    # Y3_path = 'variance/Y3_loc_min.npy'

    # Y0_path = 'variance/Y0_loc_max.npy'
    # Y1_path = 'variance/Y1_loc_max.npy'
    # Y2_path = 'variance/Y2_loc_max.npy'
    # Y3_path = 'variance/Y3_loc_max.npy'

    Y0_loc = np.load(Y0_path)
    Y1_loc = np.load(Y1_path)
    Y2_loc = np.load(Y2_path)
    Y3_loc = np.load(Y3_path)
    # print(Y0_loc.shape)
    # print(Y1_loc.shape)
    # print(Y2_loc.shape)
    # print(Y3_loc.shape)
    input_im = mpimg.imread(path).astype(float)
    demosaiced_im = flatcam.demosaiced(input_im, calib) * 255
    pil_image = Image.fromarray(demosaiced_im.astype(np.uint8))
    img_128 = pil_image.resize((128, 128))
    dct2_im = multiresolution.dct2Dimg(np.asarray(img_128)/255)
    x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)


    # reshape_size = x0.shape
    # x0 = x0.flatten()
    # x1 = x1.flatten()
    # x2 = x2.flatten()
    # x3 = x3.flatten()
    # x0[Y0_loc] = x0[Y0_loc] * k
    # x1[Y1_loc] = x1[Y1_loc] * k
    # x2[Y2_loc] = x2[Y2_loc] * k
    # x3[Y3_loc] = x3[Y3_loc] * k
    # x0 = x0.reshape(reshape_size)
    # x1 = x1.reshape(reshape_size)
    # x2 = x2.reshape(reshape_size)
    # x3 = x3.reshape(reshape_size)


    std = k
    x0, x0n, t0 = add_noise_std (x0, Y0_loc, std, 13)
    x1, x1n, t1 = add_noise_std (x1, Y1_loc, std, 25)
    x2, x2n, t2 = add_noise_std (x2, Y2_loc, std, 18)
    x3, x3n, t3 = add_noise_std (x3, Y3_loc, std, 29)
    t_2 = time.time()
    dct2_im = multiresolution.join_quarters(x0, x1, x2, x3)
    idct2_im_n = multiresolution.idct2Dimg(dct2_im) * 255
    t_a = time.time()
    idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    idct_im_pil_ns = idct_im_pil_n.resize((620, 500))
    sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil_ns), calib) * 255
    sm_pil = Image.fromarray(sensor_measurement.astype(np.uint8))
    t_3 = time.time()
    # print('noise add time ', t_1-t_0 + t0+t1+t2+t3 + t_3-t_2)
    # print('noise add time ', t_1-t_0 + t0+t1+t2+t3 + t_a-t_2)
    lmbd = 3e-4  # L2 regularization parameter
    recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    recon_images_n = Image.fromarray(recon_image.astype(np.uint8))
    recon_orig = flatcam.fcrecon(input_im, calib, lmbd) * 255
    recon_orig_pil = Image.fromarray(recon_orig.astype(np.uint8))
    psnr = metrics.peak_signal_noise_ratio(np.asarray(recon_orig_pil).astype(np.uint8),
                                          np.asarray(recon_images_n).astype(np.uint8))
    recon_images_n.save(f'std/{path.split("/")[-1].split(".")[0]}_{k}_{round(psnr,2)}.png')
    sm_pil.save (f'std/{path.split("/")[-1].split(".")[0]}_{k}_sm.png')


    t_0 = time.time()
    demosaiced_im = flatcam.demosaiced(sensor_measurement, calib) * 255
    pil_image = Image.fromarray(demosaiced_im.astype(np.uint8))
    t_00 = time.time()
    img_128 = idct2_im_n
    dct2_im = multiresolution.dct2Dimg(np.asarray(img_128)/255)
    x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)
    t_1 = time.time()

    x0, t0 = remove_noise_std(x0, Y0_loc, std, 13)
    x1, t1 = remove_noise_std(x1, Y1_loc, std, 25)
    x2, t2 = remove_noise_std(x2, Y2_loc, std, 18)
    x3, t3 = remove_noise_std(x3, Y3_loc, std, 29)

    t_2 = time.time()
    dct2_im = multiresolution.join_quarters(x0, x1, x2, x3)
    idct2_im_n = multiresolution.idct2Dimg(dct2_im) * 255
    idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    idct_im_pil_ns = idct_im_pil_n.resize((620, 500))
    sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil_ns), calib)
    t_3 = time.time()
    # print('noise remove time ', t_1 - t_0 + t0 + t1 + t2 + t3 + t_3 - t_2)
    # print('noise remove time ', t_1 - t_00 + t0 + t1 + t2 + t3 + t_3 - t_2)
    lmbd = 3e-4  # L2 regularization parameter
    recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    recon_images_n = Image.fromarray(recon_image.astype(np.uint8))
    recon_orig = flatcam.fcrecon(input_im, calib, lmbd) * 255
    recon_orig_pil = Image.fromarray(recon_orig.astype(np.uint8))
    psnrs = metrics.peak_signal_noise_ratio(np.asarray(recon_orig_pil).astype(np.uint8),
                                           np.asarray(recon_images_n).astype(np.uint8))
    recon_images_n.save(f'std/{path.split("/")[-1].split(".")[0]}_{k}_{round(psnrs, 2)}.png')

    # dct2_im = multiresolution.join_quarters(x0n, x1n, x2n, x3n)
    # idct2_im_n = multiresolution.idct2Dimg(dct2_im) *255
    # idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    # idct_im_pil_ns = idct_im_pil_n.resize((620, 500))
    # sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil_ns), calib) * 255
    # lmbd = 3e-4  # L2 regularization parameter
    # recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    # recon_images_n = Image.fromarray(recon_image.astype(np.uint8))
    # psnr = metrics.peak_signal_noise_ratio(np.asarray(recon_orig_pil).astype(np.uint8),
    #                                       np.asarray(recon_images_n).astype(np.uint8))
    # recon_images_n.save(f'{path.split("/")[-1].split(".")[0]}_{k}_{round(psnr,2)}_zn.png')
    return psnr


def find_locations(path, percent, min, max):
    var = np.load(path).astype(np.float64)
    no_of_locs = math.floor(percent * var.shape[0] / 100)
    indexs = np.where((var > min) & (var < max))[0]
    np.random.shuffle(indexs)
    locs = indexs[:no_of_locs]
    return locs


def compute_avg_psnr(data_path, std):
    src_path = os.path.join(data_path, 'rice_rand_87_testset')
    root_dir = [x for x in os.walk(src_path)]
    class_dir = root_dir[0][0]
    image_list = root_dir[0][-1]
    psnrs = []
    for image in image_list:
        img_path = os.path.join(class_dir, image)
        psnrs.append(add_noise(img_path, std))
        print(f'Done for {image} !!!')

    avg_psnr = sum(psnrs) / len(image_list)
    print(psnrs)
    return avg_psnr


if __name__ == '__main__':
    data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face'
    # variance_quarters(data_path)
    Y0_path = 'variances/Y0.npy'
    Y1_path = 'variances/Y1.npy'
    Y2_path = 'variances/Y2.npy'
    Y3_path = 'variances/Y3.npy'

    # save_hist(Y0_path)
    # save_hist(Y1_path)
    # save_hist(Y2_path)
    # save_hist(Y3_path)

    per = 5
    Y0_loc = find_locations(Y0_path, per, 0.170200719, 3.35*10**5)
    Y1_loc = find_locations(Y1_path, per, 0.11718231, 0.43602603)
    Y2_loc = find_locations(Y2_path, per, 0.12516247, 0.61263815)
    Y3_loc = find_locations(Y3_path, per, 0.140445, 0.18084206)

    # Y0_loc = find_locations_max(Y0_path, 1)
    # Y1_loc = find_locations_max(Y1_path, 1)
    # Y2_loc = find_locations_max(Y2_path, 1)
    # Y3_loc = find_locations_max(Y3_path, 1)
    # # Y0_loc = find_locations_min(Y0_path, 1)
    # # Y1_loc = find_locations_min(Y1_path, 1)
    # # Y2_loc = find_locations_min(Y2_path, 1)
    # # Y3_loc = find_locations_min(Y3_path, 1)

    np.save(f'variances/Y0_loc_{per}', Y0_loc)
    np.save(f'variances/Y1_loc_{per}', Y1_loc)
    np.save(f'variances/Y2_loc_{per}', Y2_loc)
    np.save(f'variances/Y3_loc_{per}', Y3_loc)

    # np.save('variance/Y0_loc_max', Y0_loc)
    # np.save('variance/Y1_loc_max', Y1_loc)
    # np.save('variance/Y2_loc_max', Y2_loc)
    # np.save('variance/Y3_loc_max', Y3_loc)

    # # np.save('variance/Y0_loc_min', Y0_loc)
    # # np.save('variance/Y1_loc_min', Y1_loc)
    # # np.save('variance/Y2_loc_min', Y2_loc)
    # # np.save('variance/Y3_loc_min', Y3_loc)
    paths = ['/scratch/s571b087/project/Lensless_Imaging/rice_face/fc_captures/10/115.png',
            '/scratch/s571b087/project/Lensless_Imaging/rice_face/fc_captures/07/015.png'
        , '/scratch/s571b087/project/Lensless_Imaging/rice_face/fc_captures/09/141.png',
            '/scratch/s571b087/project/Lensless_Imaging/rice_face/fc_captures/04/064.png'
        , '/scratch/s571b087/project/Lensless_Imaging/rice_face/fc_captures/15/079.png', 'orig/001.png']
    # for i in range (0,16):
    # for path in paths:
    #     add_noise(path,0.5, per)
       # os.system(f'mkdir {i}')
       # os.system(f'mv *.png {i}')
    # path = 'orig/001.png'
    # path = '/scratch/s571b087/project/Lensless_Imaging/rice_face/fc_captures/15/079.png'
    # stds = [0.5,0.6,0.7, 0.8,0.9, 1.0,1.1, 1.2]
    # for i in stds:
    #     add_noise(path, i)
#    data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face'
#    print(compute_avg_psnr(data_path, 1.2))
