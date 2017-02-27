#!/usr/bin/env python3
import cv2
from math import exp
import numpy as np

def GaussianKernel(size, sigma):
    centre = size // 2 + 1
    
    def get_coeffs(pos_x, pos_y):
        return exp(-1.0 * ((pos_x - centre)**2 +
                           (pos_y - centre)**2) / (2 * sigma**2))
    
    gaussian_filter = np.zeros((size, size))
    for pos_x in range(size):
        for pos_y in range(size):
            gaussian_filter[pos_x, pos_y] = get_coeffs(pos_x+1, pos_y+1)
    gaussian_filter /= np.sum(gaussian_filter)
    return gaussian_filter

# DoG - Difference of Gaussians
def d_o_g(k_size, sigma_1, sigma_2):
    kernel_1 = GaussianKernel(k_size, sigma_1)
    kernel_2 = GaussianKernel(k_size, sigma_2)
    return kernel_1 - kernel_2

def rescale(image):
    image = image.astype('float32')
    current_min = np.min(image)
    current_max = np.max(image)
    image = (image - current_min)/(current_max - current_min) * 255
    return image

#def compute_derivatives(image):
#    dI_x, dI_y = np.gradient(image)
#    return (dI_x, dI_y)

def compute_derivatives(image):
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    return (sobelx, sobely)

def color_image(image, indices):
    colored_image = image.copy()
    for position in indices:
        y, x = position
        colored_image[y, x, 0] = 0
        colored_image[y, x, 1] = 0
        colored_image[y, x, 2] = 255
                     
    return colored_image

def open_image(image_path):
    image = cv2.imread(image_path)
    image = np.array(image, dtype = np.float32)
    return image

def save_image(image, path):
    image = rescale(image)
    image = np.array(image, dtype = np.uint8)
    cv2.imwrite(path, image)