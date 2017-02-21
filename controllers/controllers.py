#!/usr/bin/env python3
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