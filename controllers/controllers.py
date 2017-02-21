#!/usr/bin/env python3
import cv2
from scipy.ndimage import filters
from math import exp
import matplotlib.pyplot as plt
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

def rescale(image):
    image = image.astype('float32')
    current_min = np.min(image)
    current_max = np.max(image)
    image = (image - current_min)/(current_max - current_min) * 255
    return image

def compute_derivatives(image):
    dI_x, dI_y = np.gradient(image)
    return (dI_x, dI_y)

def color_image(image, indices):
    colored_image = image.copy()
    for position in indices:
        x, y = position
        colored_image[x, y, 0] = 0
        colored_image[x, y, 1] = 0
        colored_image[x, y, 2] = 255
                     
    return colored_image
        

def harris_matrix(image, window_size, sigma):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    I_x, I_y = compute_derivatives(gray_image)
    I_xx = I_x ** 2
    I_yy = I_y ** 2
    I_xy = I_x * I_y
    
    h, w, _ = image.shape
    kernel = GaussianKernel(window_size, sigma)
    offset = window_size // 2
    corner_list = []
    
    tmp = np.zeros((h,
                    w))
    
    for x in range(offset, w - offset):
        for y in range(offset, h - offset):
            xx = I_xx[y - offset:y + offset + 1,
                      x - offset:x + offset + 1]
            yy = I_yy[y - offset:y + offset + 1,
                      x - offset:x + offset + 1]
            xy = I_xy[y - offset:y + offset + 1,
                      x - offset:x + offset + 1]
            
            # w * I - smooth 
            xx = np.multiply(kernel, xx)
            yy = np.multiply(kernel, yy)
            xy = np.multiply(kernel, xy)
            
            # sums
            Sxx = xx.sum()
            Syy = yy.sum()
            Sxy = xy.sum()
            
            # Find determinant and trace, use to get
            # corner strength function
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            c = det / (trace + 1e-8) # avoiding dividing by zero
            
            if c > 1000:
                tmp[y - offset, x - offset] = c
                corner_list.append([y - offset, x - offset, c])
            
    # filter three neighbours
    max_i = filters.maximum_filter(tmp, (3,3))
    tmp *= (tmp == max_i)
    max_x, max_y = np.nonzero(tmp)
    indices = [pos for pos in zip(max_x, max_y)]
    colored_image = color_image(image, indices)
    cv2.imwrite('result.png', colored_image)
    
    return corner_list, colored_image, tmp#, ins
    
if __name__ == '__main__':
    image = cv2.imread('../checkerboard.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    c, colored_image, tmp = harris_matrix(image,
                      5,
                      30)