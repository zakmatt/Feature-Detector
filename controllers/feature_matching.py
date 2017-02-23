#!/usr/bin/env python3
import argparse
import cv2
from harris import Harris
from feature_description import get_description
import numpy as np

def get_image_description(image, window_size, sigma, threshold, bin_size):
    harris = Harris(image, window_size, sigma, threshold)
    harris.harris_matrix()
    harris.gradient_matrix()
    description = get_description(harris, bin_size)
    return description

def match_features(features_1, features_2):
    
    def feature_distance(feature_1, feature_2):
        feature_1 = feature_1.flatten()
        feature_2 = feature_2.flatten()
        distance = np.sqrt(((feature_1 - feature_2) ** 2).sum())
        return distance
    
    best_match = np.inf
    second_best_match = np.inf
    for feature_1 in features_1:
        for feature_2 in features_2:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i_1",
                        "--image_1",
                        help="First image to match",
                        required=True)
    parser.add_argument("-i_2",
                        "--image_2",
                        help="Second image to match",
                        required=True)
    parser.add_argument("-w",
                        "--window_size",
                        help="Window size",
                        required=True)
    parser.add_argument("-s",
                        "--sigma",
                        help="Sigma value",
                        required=True)
    parser.add_argument("-t",
                        "--threshold",
                        help="Threshold value",
                        required=True)
    parser.add_argument("-b",
                        "--bin_size",
                        help="Size of the bin",
                        required=True)
    args = parser.parse_args()
    image_1 = args.image_1
    image_2 = args.image_2
    window_size = int(args.window_size)
    sigma = int(args.sigma)
    threshold = float(args.threshold)
    bin_size = int(args.bin_size)
    
    image_1 = cv2.imread(image_1)
    image_2 = cv2.imread(image_2)
    description_1 = get_image_description(
            image_1, window_size, sigma, threshold, bin_size
            )
    print(description_1.shape)
    description_2 = get_image_description(
            image_2, window_size, sigma, threshold, bin_size
            )
    print(description_2.shape)