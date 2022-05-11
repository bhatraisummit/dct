import cv2
import imageio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy.io import loadmat
from skimage import metrics

import flatcam
import multiresolution
import utils


def compute_avg_psnr(data_path):
    src_path = os.path.join(data_path, 'rice_rand_87')
    root_dir = [x for x in os.walk(src_path)]
    class_dir = root_dir[0][0]
    image_list = root_dir[0][-1]
    psnrs = []
    for image in image_list:
        img_path = os.path.join(class_dir, image)
        psnrs.append(psnr_singl_img(img_path, image))
        print(f'Done for {image} !!!')

    avg_psnr = sum(psnrs) / len(image_list)
    print(psnrs)
    return avg_psnr


def psnr_singl_img(img_path, img_name):
    img_dir_path = f'images/{img_name.split(".")[0]}'
    if not os.path.exists (img_dir_path):
        os.makedirs (img_dir_path)
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    flatcam.clean_calib(calib)
    input_im = mpimg.imread(img_path).astype(float)
    demosaiced_im = flatcam.demosaiced(input_im, calib) * 255
    pil_image = Image.fromarray(demosaiced_im.astype(np.uint8))
    pil_image.save(f'{img_dir_path}/pil_image.png')
    img_128 = pil_image.resize((128, 128))
    img_128.save(f'{img_dir_path}/img_128.png')
    dct2_im = multiresolution.dct2Dimg(np.asarray(img_128) / 255)
    x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)
    x0_n = utils.add_noise_percent(x0, 0.2, 1, 10, 15)
    x1_n = utils.add_noise_percent(x1, 0.2, 5, 13, 23)
    x2_n = utils.add_noise_percent(x2, 0.2, 5, 8, 42)
    x3_n = utils.add_noise_percent(x3, 0.2, 10, 53, 65)
    dct2_im_n = multiresolution.join_quarters(x0_n, x1_n, x2_n, x3_n)
    idct2_im_n = multiresolution.idct2Dimg(dct2_im_n) * 255
    idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    idct_im_pil_n.save(f'{img_dir_path}/idct_im_pil_n.png')
    idct_im_pil_ns = idct_im_pil_n.resize((620, 500))
    idct_im_pil_ns.save(f'{img_dir_path}/idct_im_pil_n_620_500.png')
    sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil_ns), calib) * 255
    sensor_measurements = Image.fromarray(sensor_measurement.astype(np.uint8))
    sensor_measurements.save(f'{img_dir_path}/sensor_measurement_n.png')
    lmbd = 3e-4  # L2 regularization parameter
    recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    recon_images_n = Image.fromarray(recon_image.astype(np.uint8))
    recon_images_n.save(f'images/{img_name}')

    recon_orig = flatcam.fcrecon(input_im, calib, lmbd) * 255
    recon_orig_pil = Image.fromarray(recon_orig.astype(np.uint8))
    recon_orig_pil.save(f'{img_dir_path}/recon_orig.png')
    return metrics.peak_signal_noise_ratio(np.asarray(recon_orig_pil).astype(np.uint8),
                                          np.asarray(recon_images_n).astype(np.uint8))

    # Remove noise and reconstruct
    # x0 = utils.remove_noise_percent(x0_n, 0.2, 1, 10, 15)
    # x1 = utils.remove_noise_percent(x1_n, 0.2, 5, 13, 23)
    # x2 = utils.remove_noise_percent(x2_n, 0.2, 5, 8, 42)
    # x3 = utils.remo_noise_percent(x3_n, 0.2, 10, 53, 65)
    # dct2_im_wo_n = multiresolution.join_quarters(x0, x1, x2, x3)
    # idct2_im_wo_n = multiresolution.idct2Dimg(dct2_im_wo_n) * 255
    # idct_im_pil_wo_n = Image.fromarray(idct2_im_wo_n.astype(np.uint8))
    # idct_im_pil_wo_n.save('images/idct_im_pil_wo_n.png')
    # idct_im_pils_wo_ns = idct_im_pil_wo_n.resize((620, 500))
    # idct_im_pils_wo_ns.save('images/idct_im_pil_wo_n_620_500.png')
    # sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pils_wo_ns), calib) * 255
    # sensor_measurements = Image.fromarray(sensor_measurement.astype(np.uint8))
    # sensor_measurements.save('images/sensor_measurement_wo_n.png')
    # lmbd = 3e-4  # L2 regularization parameter
    # recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    # recon_images = Image.fromarray(recon_image.astype(np.uint8))
    # recon_images.save('images/recon_image_wo_n.png')
    # print(metrics.peak_signal_noise_ratio(np.asarray(recon_orig_pil).astype(np.uint8), np.asarray(recon_images).astype(np.uint8)))


if __name__ == '__main__':
    data_path = '/scratch/s571b087/project/Lensless_Imaging/rice_face'
    singl_img = '/scratch/s571b087/project/Lensless_Imaging/rice_face/rice_rand_87/43_080.png'
    # psnr_singl_img(singl_img)
    print(compute_avg_psnr(data_path))
