import os
import numpy as np
import matplotlib.pyplot as plt
from utils import imread
from detectBlobs import detectBlobs
from drawBlobs import drawBlobs

# Evaluation code for blob detection
# Your goal is to implement scale space blob detection using LoG  
#
# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji

images = ['butterfly.jpg','einstein.jpg','faces.jpg','fishes.jpg','football.jpg','sunflowers.jpg']
for imageName in images:
    numBlobsToDraw = 1000
    imName = imageName.split('.')[0]
    datadir = os.path.join('..', 'data', 'blobs')
    im = imread(os.path.join(datadir, imageName))
    blobs = detectBlobs(im)  
    drawBlobs(im, blobs, numBlobsToDraw)