import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct, idct
from PIL import Image
from scipy.io import loadmat
import flatcam


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


def multiresolution_dct_test(dct2_img):
    img_64 = dct2_img.resize((64, 64))
    img_32 = dct2_img.resize((32, 32))
    img_dct_64 = dct2Dimg(np.asarray(img_64)).astype(np.uint8)
    image_64 = Image.fromarray(img_dct_64)
    image_64.save('dct_64.png')
    image_32 = image_64.resize((32, 32))
    img_dct_32 = np.asarray(image_32)
    img_dct_32 = dct2Dimg(np.asarray(img_32)).astype(np.uint8)
    image_32 = Image.fromarray(img_dct_32)
    image_32.save('dct_32.png')
    x0, x1, x2, x3 = divide_dct(img_dct_64)
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(x0)
    plt.axis('off')
    plt.title('X0')
    plt.subplot(2, 3, 4)
    plt.imshow(x1)
    plt.axis('off')
    plt.title('X1')
    plt.subplot(2, 3, 2)
    plt.imshow(x2)
    plt.axis('off')
    plt.title('X2')
    plt.subplot(2, 3, 5)
    plt.imshow(x3)
    plt.axis('off')
    plt.title('X3')
    plt.subplot(2, 3, 3)
    plt.imshow(img_dct_64)
    plt.axis('off')
    plt.title('img_64')
    plt.subplot(2, 3, 6)
    plt.imshow(img_dct_32)
    plt.axis('off')
    plt.title('img_32')
    plt.show()
    image_x0 = Image.fromarray(x0.astype(np.uint8))
    image_x0.save('x0.png')
    image_x1 = Image.fromarray(x1.astype(np.uint8))
    image_x1.save('x1.png')
    image_x2 = Image.fromarray(x2.astype(np.uint8))
    image_x2.save('x2.png')
    image_x3 = Image.fromarray(x3.astype(np.uint8))
    image_x3.save('x3.png')
    return np.concatenate((x0, x1, x2, x3, img_dct_32), axis=2)


def tests():
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    senson_image = Image.open('bayer_rgb.png')
    multiresolution_dct_test(senson_image)


if __name__ == '__main__':
    tests()
