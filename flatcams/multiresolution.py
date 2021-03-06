import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image


# Normalize Data
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())


# dct image
def dct2(pix):
    return dct(dct(pix, axis=0, norm='ortho'), axis=1, norm='ortho')


def dct2Dimg(img):
    dct_R = dct2(img[:, :, 0])
    dct_G = dct2(img[:, :, 1])
    dct_B = dct2(img[:, :, 2])
    return np.dstack([dct_R, dct_G, dct_B])


# inverse dct2
def idct2(pix):
    return idct(idct(pix, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2Dimg(dct_img):
    idct_R = idct2(dct_img[:, :, 0])
    idct_G = idct2(dct_img[:, :, 1])
    idct_B = idct2(dct_img[:, :, 2])
    return np.dstack([idct_R, idct_G, idct_B])


def divide_dct(dct2_img):
    height = dct2_img.shape[0]
    width = dct2_img.shape[1]
    mid_height = height // 2
    mid_width = width // 2
    x0 = dct2_img[:mid_height, :mid_width]
    x1 = dct2_img[mid_height:height, :mid_width]
    x2 = dct2_img[:mid_height, mid_width:width]
    x3 = dct2_img[mid_height:height, mid_width:width]
    return x0, x1, x2, x3

def join_quarters(x0, x1, x2, x3):
    height = x0.shape[0] * 2
    width = x0.shape[1] * 2
    dct2_img = np.zeros((height, width, 3))
    mid_height = height // 2
    mid_width = width // 2
    dct2_img[:mid_height, :mid_width] = x0
    dct2_img[mid_height:height, :mid_width] = x1
    dct2_img[:mid_height, mid_width:width] = x2
    dct2_img[mid_height:height, mid_width:width] = x3
    return dct2_img


# resize and then dct2
def multiresolution_dct(dct2_img):
    img_64 = dct2_img.resize((64, 64))
    img_dct_64 = dct2Dimg(np.asarray(img_64)).astype(np.uint8)
    img_128 = dct2_img.resize((128, 128))
    img_dct_128 = dct2Dimg(np.asarray(img_128)).astype(np.uint8)
    x0, x1, x2, x3 = divide_dct(img_dct_128)
    return np.concatenate((x0, x1, x2, x3, img_dct_64), axis=2)

