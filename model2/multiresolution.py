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


# resize and then dct2
def multiresolution_dct(dct2_img):
    img_64 = dct2_img.resize((64, 64))
    img_dct_64 = dct2Dimg(np.asarray(img_64)).astype(np.uint8)
    img_32 = dct2_img.resize((32, 32))
    img_dct_32 = dct2Dimg(np.asarray(img_32)).astype(np.uint8)
    x0, x1, x2, x3 = divide_dct(img_dct_64)
    return np.concatenate((x0, x1, x2, x3, img_dct_32), axis=2)

