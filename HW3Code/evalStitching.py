import numpy as np
import matplotlib.pyplot as plt
import os
from utils import imread
from utils import showMatches
from detectBlobs2 import detectBlobs
from drawBlobs2 import drawBlobs
from computeSift import compute_sift
from computeMatches import computeMatches
from ransac import ransac
from mergeImages import mergeImages
import random
import warnings
warnings.filterwarnings("ignore")

#Image directory
dataDir = os.path.join('..', 'data', 'stitching')

#Read input images
testExamples = ['stop', 'car', 'building', 'book', 'house1', 'house2', 'kitchen', 'park', 'pier', 'roof', 'table','eg']

for exampleIndex in range(len(testExamples)):
    random.seed(1)
    imageName1 = '{}_1.jpg'.format(testExamples[exampleIndex])
    imageName2 = '{}_2.jpg'.format(testExamples[exampleIndex])

    im1 = imread(os.path.join(dataDir, imageName1))
    im2 = imread(os.path.join(dataDir, imageName2))

    #Detect keypoints
    blobs1 = detectBlobs(im1)
    blobs2 = detectBlobs(im2)

    #Compute SIFT features 
    sift1 = compute_sift(im1, blobs1[:, 0:4])
    sift2 = compute_sift(im2, blobs2[:, 0:4])

    #Display SIFT features 
    t0 = "SIFT_Features:"+imageName1
    t1 = "SIFT_Features:"+imageName2
    drawBlobs(im1, blobs1, t0)
    drawBlobs(im2, blobs2, t1)
    
    matches = computeMatches(sift1, sift2)
    t2 = "Ratio_Matching:"+str(testExamples[exampleIndex])
    showMatches(im1, im2, blobs1, blobs2, matches, t2)

    #Ransac to find correct matches and compute transformation
    inliers, transf = ransac(matches, blobs1, blobs2)

    goodMatches = np.full_like(matches, -1)
    goodMatches[inliers] = matches[inliers]

    t3 = "RANSAC:"+str(testExamples[exampleIndex])
    showMatches(im1, im2, blobs1, blobs2, goodMatches, title=t3)

    #Merge two images and display the output
    stitchIm = mergeImages(im2, im1, transf)
    plt.figure()
    plt.imshow(stitchIm)
    plt.title('Stitched Image: {}'.format(testExamples[exampleIndex]))
    plt.savefig('StitchedImage:{}.png'.format(testExamples[exampleIndex]))
    #files.download('StitchedImage:{}.png'.format(testExamples[exampleIndex]))
    plt.show()