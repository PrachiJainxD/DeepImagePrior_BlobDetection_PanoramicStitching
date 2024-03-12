import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """
    N, d = np.shape(f1)
    M, d = np.shape(f2)    
    
    matches = np.ones((N, )) * -1

    method = 0
    if method == 0:    
        print("Computing matching using SSD")
        SSD = cdist(f1, f2, 'seuclidean')
        #For each feature from image 1, find its nearest match image 2 and store indices of lowest/minima SSD feature
        matches = np.argmin(SSD, axis=1)

    elif method == 1:
        print("Computing matching using Ratio:")
        distanceSSD = cdist(f1, f2, 'seuclidean')
        #For each Descriptor in the First Image, Select its Match to Second Image        
        bestMatch = list(np.argmin( distanceSSD, axis=1))
        for x in range(N):
        #For each Descriptor in the First Image, Select its Match to Second Imag          
            bestId = bestMatch[x]
            bestValue = distanceSSD[x, bestId] #Most likely nearest neighbor match

            #Next most likely match after best one
            secondBestMatch = list(distanceSSD[x,:])
            secondBestMatch[bestId] = float('inf')
            secondBestValue = min(secondBestMatch)
           # After determining best and second best match, compute ratio
            if bestId != -1 and bestValue/secondBestValue <= 0.8:
                matches[x] = bestId
            else:
                matches[x] = -1
    return matches 

