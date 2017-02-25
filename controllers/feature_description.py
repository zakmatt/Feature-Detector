#!/usr/bin/env python3
import cv2
from harris import Harris
import numpy as np

def get_description(harris, bin_size):
    
    
    
    def create_histogram(neighbour_gradient_matrix,
                         gradient_matrix,
                         bin_size,
                         p,
                         q):
        #pi = np.pi
        # vector of 8 bins
        histogram = np.zeros(8)
        #d_x,
        #d_y,
        #LA.norm([d_x, d_y]),
        #cv2.fastAtan2(d_y, d_x)
        h, w, _ = neighbour_gradient_matrix.shape
        #print(h, w)
        for y in range(h):
            for x in range(w):
                element = neighbour_gradient_matrix[y, x]
                bin_position = int(element[3] / 45) % 8
                histogram[bin_position] += element[2]
        #histogram = histogram / np.sqrt(np.sum(histogram ** 2))
        max_value = np.max(histogram)
        min_value = np.min(histogram)
        histogram = [(current_bin - min_value)/(max_value - min_value) for current_bin in histogram]
        #print(histogram)
        
        '''
        weights_sum = np.array([[neighbour_gradient_matrix[y, x][2] for x in range(bin_size)]
                        for y in range(bin_size)]).sum()
        
        weights_sum += 1e-8
        
        weights = np.array([[neighbour_gradient_matrix[y, x][2] for x in range(bin_size)]
                        for y in range(bin_size)]) / weights_sum
        corner_theta = (gradient_matrix[p, q])[3]
        thetas = np.array([[neighbour_gradient_matrix[y, x][3] - corner_theta 
                            for x in range(bin_size)] for y in range(bin_size)])
        bins = [i*0.25*pi for i in range(9)]
        # index 0 is an array with an amount of elements
        # in each bin
        histogram = np.histogram(thetas, bins = bins, weights = weights)[0]
        '''
        return histogram
        
    corner_list = harris.corner_list
    gradient_matrix = harris.gradient_matrix
    all_neighbourhoods = []
    for corner in corner_list:
        y, x, c = corner
        nearest_neighbourhood = []
        # take neighbours from [-bin;bin] range in
        # y and x axes
        for p in range(-2 * bin_size, 2 * bin_size, bin_size):
            for q in range(-2 * bin_size, 2 * bin_size, bin_size):
                nearest_neighbourhood.append(
                        create_histogram(
                                gradient_matrix[
                                        y + p:y + p + bin_size,
                                        x + q:x + q + bin_size
                                        ],
                                gradient_matrix,
                                bin_size,
                                y + p + bin_size / 2,
                                x + q + bin_size / 2
                                )
                        )
                #break
            #break
        #break
        all_neighbourhoods.append(nearest_neighbourhood)
    return np.array(all_neighbourhoods)

if __name__ == '__main__':
    #image = cv2.imread('../checkerboard.png')
    image = cv2.imread('img1.ppm')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = Harris(image, 5, 30, 2500)
    harris.harris_matrix()
    harris.gradient_matrix()
    all_neighbourhoods = get_description(harris, 4)
    #print(all_neighbourhoods[0,0])