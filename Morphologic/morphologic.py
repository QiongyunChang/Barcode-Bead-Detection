'''
(1) Gaussian Blur
(2) Adaptive threshold
(3) Opening : Erosion -> dilation
'''

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def GaussianFunction(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def GaussianKernel(KernelSize):
    sigma = 0.3 * ((KernelSize - 1) * 0.5 - 1) + 0.8
    half = round(KernelSize / 2)
    i, j = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = GaussianFunction(i, j, sigma)
    total = 0  # 各係數相加後總值
    total = np.sum(kernel)

    return np.array(kernel) / total

# Convolution

def image_padding(image, KernelSize):
    p = int(np.floor(KernelSize / 2))
    # print(p)
    (w, h) = np.shape(image)
    pad_img = np.zeros((w + 2 * p, h + 2 * p))
    pad_img[int(p):int(-1 * p), int(p):int(-1 * p)] = image
    return pad_img


def convolution(image, Kernel, stride=1, padding=True):
    # upside down the kernel to meet the requirement of convolution
    # note that cross cross-correlation does not need rotate the kernal
    Kernel = np.flip(Kernel.T, axis=0)
    KernelSize = np.shape(Kernel)[0]
    row, col = image.shape
    # print(KernelSize)
    # print('image',np.shape(image))
    if stride is None:
        stride = KernelSize
    if padding:
        pad_img = image_padding(image, KernelSize)
        resx = np.zeros((row, col))
    else:
        pad_img = image
        resx = np.zeros((int((row - KernelSize) / stride) + 1, int((col - KernelSize) / stride) + 1))

    # print('stride',stride)
    nrow, ncol = pad_img.shape
    # print('nrow',nrow,'ncol',ncol)

    xpatch = np.arange(0, nrow - KernelSize + 1, stride)
    ypatch = np.arange(0, ncol - KernelSize + 1, stride)

    for x_id, x in enumerate(xpatch):
        for y_id, y in enumerate(ypatch):
            matrix = pad_img[x:x + KernelSize, y:y + KernelSize]
            multi_matrix = np.multiply(matrix, Kernel)
            resx[x_id, y_id] = np.sum(multi_matrix)
    return resx


def Gaussian_Filter(KernelSize, image):
    # create gaussian kernel
    kernel = GaussianKernel(KernelSize)
    resx = convolution(image, kernel)
    resx = resx.astype(np.uint8)
    return resx

 # Dilation
def dilation(img2):
    imgDilate = np.zeros((height, width), dtype=np.uint8)
    # Define the structuring element
    SED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    constant1 = 1
    for i in range(constant1, height - constant1):
        for j in range(constant1, width - constant1):
            temp = img2[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]
            product = temp * SED
            imgDilate[i, j] = np.max(product)
    return imgDilate

 # Erosion
def erosion(img1):
    k = 5
    SE = np.ones((k, k), dtype=np.uint8)
    constant = (k - 1) // 2
    imgErode = np.zeros((height, width), dtype=np.uint8)
    for i in range(constant, height - constant):
        for j in range(constant, width - constant):
            temp = img1[i - constant:i + constant + 1, j - constant:j + constant + 1]
            product = temp * SE
            imgErode[i, j] = np.min(product)
    return imgErode

#  Global threshold
def thresholding(image):
    image[image > 127] = 255
    image[image != 255] = 0
    return  image
#  Adaptive threshold
def adaptive_thresh(input_img):
    h, w = input_img.shape
    S = w/8
    s2 = S/2
    T = 15.0
    #integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row,col] = input_img[0:row,0:col].sum()
    #output img
    out_img = np.zeros_like(input_img)

    for col in range(w):
        for row in range(h):
            #SxS region
            y0 = max(row-s2, 0)
            y1 = min(row+s2, h-1)
            x0 = max(col-s2, 0)
            x1 = min(col+s2, w-1)

            count = (y1-y0)*(x1-x0)

            sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]

            if input_img[row, col]*count < sum_*(100.-T)/100.:
                out_img[row,col] = 0
            else:
                out_img[row,col] = 255

    return out_img


img = cv2.imread('W_A3_0_3.jpg', 0)
# print(img_org.shape)
KernelSize = 3
height, width = img.shape
# noise
img = Gaussian_Filter(KernelSize,img)
# a = thresholding(img)
a= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,63,14)
erosion_img = erosion(a)
opening_img = dilation(erosion_img)  # erosion -> dilation
plt.imshow(opening_img, cmap='gray')
plt.show()
