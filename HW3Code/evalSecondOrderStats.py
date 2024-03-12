# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2020
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 3

import numpy as np
import skimage.io as sio
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#Imported librarires
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def second_order_statistics(ix, iy, ws):
    height, width = np.shape(ix)
    R = np.zeros((height, width))

    k = 0.03    
    
    IXIX = gaussian_filter(np.multiply(ix, ix),(ws-1)//2)
    IXIY = gaussian_filter(np.multiply(ix, iy),(ws-1)//2)
    IYIY = gaussian_filter(np.multiply(iy, iy),(ws-1)//2)
    
    #For each pixel(u,v)
    for u in range(height):
        for v in range(width):
            SecondMoment = np.zeros((2,2))
            a = IXIX[u, v] 
            b = IXIY[u,v] 
            c = b 
            d = IYIY[u, v] 
            SecondMoment[0, 0] = a 
            SecondMoment[0, 1] = b 
            SecondMoment[1, 0] = c 
            SecondMoment[1, 1] = d
            #Compute quantity R
            lambda1 = a * d
            lambda2 =  b * c
            sumlamda12 = a + d
            R[u, v] = (lambda1 - lambda2) - k *(np.power(sumlamda12, 2))

    return R

color_list = ['white','darkblue']
cmap = LinearSegmentedColormap.from_list("",color_list)
files = ["cones_im2.png","teddy_im2.png"]

for image_file in files:
    # read image
    image = sio.imread(image_file)
    #convert image to gray
    image = rgb2gray(image)
    #compute differentiations along x and y axis respectively
    # x-diff
    #--------- add your code here ------------------#
    kernelX = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)
    Ix = ndimage.convolve(image, kernelX)

    # y-diff
    #--------- add your code here ------------------#
    kernelY = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], np.float32)
    Iy = ndimage.convolve(image, kernelY)

    #set window size
    #--------- modify this accordingly ------------------#
    for ws in range(5, 11, 5):
        heatMapImg = second_order_statistics(Ix, Iy, ws)
        plt.imshow(heatMapImg, cmap =cmap)
        plt.title("{} Heat Map with radius:{}".format(image_file, ws))
        plt.show()
        plt.savefig("{}_{}HeatMap.png".format(image_file, ws), cmap =cmap)