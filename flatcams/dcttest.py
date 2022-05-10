import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import imageio
import cv2
from PIL import Image
from skimage import metrics
import utils

import flatcam
import multiresolution


# Load data
def test1():
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    flatcam.clean_calib(calib)
    input_im = mpimg.imread('001.png').astype(float)
    demosaiced_im = flatcam.demosaiced(input_im, calib)
    print(demosaiced_im.min(), demosaiced_im.max())
    demosaiced_img = flatcam.demosaiced(input_im, calib) * 255
    print(demosaiced_img.min(), demosaiced_img.max())
    pil_image = Image.fromarray(demosaiced_img.astype(np.uint8))
    pil_image.save('pil_image.png')
    img_128 = pil_image.resize((128, 128))
    img_128.save('img_128.png')
    dct_im = multiresolution.dct2Dimg(np.asarray(img_128) / 255)
    print('dct_im ', dct_im.min(), dct_im.max())
    dctimg = dct_im * 255
    dct_im_pil = Image.fromarray(dctimg.astype(np.uint8))
    dct_im_pil.save('dct_im_pil.png')
    idct_im = multiresolution.idct2Dimg(dct_im) * 255
    # idct_im = multiresolution.normalize_data(idct_im) * 255
    print(idct_im.min(), idct_im.max())
    idct_im_pil = Image.fromarray(idct_im.astype(np.uint8))
    idct_im_pil.save('idct_im_pil.png')
    idct_im_pils = idct_im_pil.resize((620, 500))
    idct_im_pils.save('idct_im_pil_620_500.png')
    # works when multiplied by 255 with unnormalized
    sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil), calib) * 255
    sensor_measurements = Image.fromarray(sensor_measurement.astype(np.uint8))
    sensor_measurements.save('sensor_measurement.png')
    lmbd = 3e-4  # L2 regularization parameter
    recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    recon_images = Image.fromarray(recon_image.astype(np.uint8))
    recon_images.save('recon_image.png')
    recon_orig = flatcam.fcrecon(input_im, calib, lmbd) * 255
    recon_orig_pil = Image.fromarray(recon_orig.astype(np.uint8))
    recon_orig_pil.save('recon_orig.png')
    imageio.imsave('recon_001.png', recon_orig.astype(np.uint8))
    simulate_im = flatcam.simulate(recon_orig, calib)
    sm_orig = Image.fromarray(simulate_im.astype(np.uint8))
    sm_orig.save('sm_orig_pil.png')
    imageio.imsave('sm_orig.png', simulate_im)
    recon_a_simul = flatcam.fcrecon(simulate_im, calib, lmbd) * 255
    print(metrics.peak_signal_noise_ratio(np.asarray(recon_orig_pil).astype(np.uint8), np.asarray(recon_a_simul).astype(np.uint8)))
    print(metrics.peak_signal_noise_ratio(np.asarray(recon_orig_pil).astype(np.uint8), np.asarray(recon_images).astype(np.uint8)))
    print(metrics.peak_signal_noise_ratio(np.asarray(recon_a_simul).astype(np.uint8), np.asarray(recon_images).astype(np.uint8)))
    print(metrics.peak_signal_noise_ratio(np.asarray(recon_a_simul).astype(np.uint8), np.asarray(recon_images).astype(np.uint8)))
    imageio.imsave('recon_a_simul.png', recon_a_simul.astype(np.uint8))


def test2():
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    flatcam.clean_calib(calib)
    input_im = mpimg.imread('001.png').astype(float)
    demosaiced_im = flatcam.demosaiced(input_im, calib) * 255
    pil_image = Image.fromarray(demosaiced_im.astype(np.uint8))
    pil_image.save('pil_image.png')
    img_128 = pil_image.resize((128, 128))
    img_128.save('img_128.png')
    dct2_im = multiresolution.dct2Dimg(np.asarray(img_128) / 255)
    idct_im = multiresolution.idct2Dimg(dct2_im) * 255
    x0, x1, x2, x3 = multiresolution.divide_dct(dct2_im)
    x0_n = utils.add_noise_percent(x0, 0.2, 1, 10, 15)
    x1_n = utils.add_noise_percent(x1, 0.2, 5, 13, 23)
    x2_n = utils.add_noise_percent(x2, 0.2, 5, 8, 42)
    x3_n = utils.add_noise_percent(x3, 0.2, 10, 53, 65)
    dct2_im_n = multiresolution.join_quarters(x0_n, x1_n, x2_n, x3_n)
    idct2_im_n = multiresolution.idct2Dimg(dct2_im_n) * 255
    idct_im_pil_n = Image.fromarray(idct2_im_n.astype(np.uint8))
    idct_im_pil_n.save('idct_im_pil_n.png')
    idct_im_pil_ns = idct_im_pil_n.resize((620, 500))
    idct_im_pil_ns.save('idct_im_pil_n_620_500.png')
    sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pil_ns), calib) * 255
    sensor_measurements = Image.fromarray(sensor_measurement.astype(np.uint8))
    sensor_measurements.save('sensor_measurement_n.png')
    lmbd = 3e-4  # L2 regularization parameter
    recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    recon_images_n = Image.fromarray(recon_image.astype(np.uint8))
    recon_images_n.save('recon_image_n1.png')

    # Remove noise and reconstruct
    x0 = utils.remove_noise_percent(x0_n, 0.2, 1, 10, 15)
    x1 = utils.remove_noise_percent(x1_n, 0.2, 5, 13, 23)
    x2 = utils.remove_noise_percent(x2_n, 0.2, 5, 8, 42)
    x3 = utils.remove_noise_percent(x3_n, 0.2, 10, 53, 65)
    dct2_im_wo_n = multiresolution.join_quarters(x0, x1, x2, x3)
    idct2_im_wo_n = multiresolution.idct2Dimg(dct2_im_wo_n) * 255
    idct_im_pil_wo_n = Image.fromarray(idct2_im_wo_n.astype(np.uint8))
    idct_im_pil_wo_n.save('idct_im_pil_wo_n.png')
    idct_im_pils_wo_ns = idct_im_pil_wo_n.resize((620, 500))
    idct_im_pils_wo_ns.save('idct_im_pil_wo_n_620_500.png')
    sensor_measurement = flatcam.demosaic_fc(np.asarray(idct_im_pils_wo_ns), calib) * 255
    sensor_measurements = Image.fromarray(sensor_measurement.astype(np.uint8))
    sensor_measurements.save('sensor_measurement_wo_n.png')
    lmbd = 3e-4  # L2 regularization parameter
    recon_image = flatcam.fcrecon(sensor_measurement, calib, lmbd) * 255
    recon_images = Image.fromarray(recon_image.astype(np.uint8))
    recon_images.save('recon_image_wo_n.png')
    recon_orig = flatcam.fcrecon(input_im, calib, lmbd) * 255
    recon_orig_pil = Image.fromarray(recon_orig.astype(np.uint8))
    recon_orig_pil.save('recon_orig.png')
    print(metrics.peak_signal_noise_ratio(np.asarray(recon_orig_pil).astype(np.uint8), np.asarray(recon_images_n).astype(np.uint8)))
    print(metrics.peak_signal_noise_ratio(idct_im.astype(np.uint8), np.asarray(idct_im_pil_wo_n).astype(np.uint8)))


test2()

