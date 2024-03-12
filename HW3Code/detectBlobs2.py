# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import numpy as np
import cv2
import scipy
from skimage.color import rgb2gray

#Imported libraries
from scipy.ndimage.filters import rank_filter
from itertools import chain

def  LoGFilter(initSigma, filterSize):
    #Laplacian of Gaussian Filter
    sigma = initSigma ** 2

    # Ensure right - odd filter size
    if filterSize % 2 == 0:
        filterSize += 1     
   
    idRange = np.linspace(-(filterSize - 1) / 2., (filterSize - 1) / 2., filterSize)
    x, y = np.meshgrid(idRange, idRange)
    
    kernel = np.exp(-(np.square(x) + np.square(y)) / (2. * sigma))
    kernel[kernel < np.finfo(float).eps * np.amax(kernel)] = 0
    kSum = np.sum(kernel)
    if kSum != 0:
        kernel /= np.sum(kernel)
    
    tmp = np.multiply(kernel, np.square(x) + np.square(y) - 2 * sigma) / (sigma ** 2)
    kernel = tmp - np.sum(tmp) / (filterSize ** 2)
    return kernel

def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 5 array with blob in each row in (x, y, radius, angle, score)
    
    #Convert image to grayscale and then to double such that minimum and maximum intensity are 0 and 1 respectively
    im = rgb2gray(im[:,:,:])
    im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    #Compute scale space representation
    initSigma = 1.3 #Initial Scale
    factor = 1.25 #np.sqrt(2) #Factor by which scale is mulitplied each time  
    threshold =  0.005
    finalSigma = np.power(factor, 16) #Final Scale
    
    #Dynamically decide iteration levels from initial scale, last scale and multiplication factor
    n = 12 #int(np.ceil((np.log(finalSigma) - np.log(initSigma))/np.log(factor))) #Iterations of Laplacian Scale Space   

    sigmas = np.empty(n)
    sigmas[0] = initSigma

    height, width = np.shape(im)   
    scaleSpace = np.empty((height, width, n)) 

    method = 0
    #####################################################################
    # Compute blob response (Squared Laplacian response in scale-space) #
    #####################################################################
    
    if method == 0:
        #Increase filter size, keep image same
        for i in range(n):  
            #Generate the Laplacian of Gaussian for Scale Level
            filterSize = int(round(6 * sigmas[i])) #Compute filter kernel size from previous sigma
            LoG = np.power(sigmas[i],2) * LoGFilter(sigmas[i], filterSize) # Obtain filter with normalization and apply convolution.
            nextFilter= cv2.filter2D(im, cv2.CV_32F, LoG) #Create a replica of previous filter
            scaleSpace[:,:, i] = np.square(nextFilter) #Store squared filter at each level 

            #Update sigma for next level
            if (i+1 < n):
                sigmas[i+1] = sigmas[i] * factor

    ###################################################
    # Perform Non-Maximum Suppression in Scale Space  #
    ###################################################
    maxScaleSpace = np.empty((height, width, n)) #Non-Maximum Suppression scale space for each level

    #Non-Maximum Suppression in each 2D slice
    #A point is Local Maximum if it is higher than all its neighbors (each point has 26 neighbors in 3D)    
    for level in range(n):
        maxScaleSpace[:,:,level] = rank_filter(scaleSpace[:,:, level], -1, (3,3)) #Determining Local Maximum point using rank filter
    
    layerScaleSpace = np.zeros((maxScaleSpace.shape)) #Non-Maximum Suppression scale space for all levels

    #Compute non-max suppression across all layers
    for i in range(height):
        for j in range(width):
            maxValue = np.amax(maxScaleSpace[i,j,:])
            maxIndex = np.argmax(maxScaleSpace[i,j,:])
            layerScaleSpace[i,j,maxIndex] = maxValue

    #Zero Out All Positions that are not Local Maxima of the score (if the Value if not Greater than all its Neighbors)
    for level in range(n):
        layerScaleSpace[:,:,level] = np.where((layerScaleSpace[:,:,level] == scaleSpace[:,:,level]), layerScaleSpace[:,:,level], 0)

    #Mask filter to eliminate boundaries noises
    #check
    mask = np.zeros((height, width, n))
    for i in range(n):
        boundary = int(np.ceil(sigmas[i] * np.sqrt(2))) 
        mask[boundary+1:height-boundary, boundary+1:width-boundary] = 1

    centerX = [] 
    centerY = []
    radii = []
    score = []
    
    #Set a Threshold on Squared Laplacian Response above which to report Region Detections
    for level in range(n):
        layerScaleSpace[:,:,level] = np.where((layerScaleSpace[:,:,level] > threshold) & (mask[:,:,level] == 1), layerScaleSpace[:,:,level], 0)
        
        #Obtain X and Y center coordinates for local maxima
        centerX.append(list(np.where(layerScaleSpace[:,:, level] != 0)[0]))
        centerY.append(list(np.where(layerScaleSpace[:,:,level] != 0)[1]))
        radius = np.sqrt(2) * sigmas[level] * np.ones(len(centerX[level])) #np.sqrt(2) * np.power(factor, level-1) * initSigma * np.ones(len(centerX[level])) # Compute radius
        radii.append(list(radius)) 
        score.append(layerScaleSpace[np.where(layerScaleSpace[:,:, level] != 0)[0],np.where(layerScaleSpace[:,:, level] != 0)[1],level])
    
    centerX = list(chain.from_iterable(centerX))
    centerY = list(chain.from_iterable(centerY))
    radii = list(chain.from_iterable(radii))    
    score = list(chain.from_iterable(score))   

    numBlob = len(centerX)
    blobs = []
    angle = np.zeros(numBlob)
    if (numBlob > 0):
        for i in range(numBlob):
            blobs.append([centerY[i], centerX[i], radii[i], angle[i], score[i]])
    
    return np.array(blobs)