import numpy as np
import cv2
import time
import random

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#


def calaculateError(Expected, Actual):
    xE, yE = Expected[0], Expected[1]
    xT, yT = Actual[0] , Actual[1]
    error = np.sqrt((xE-xT)**2 + (yE-yT)**2)
    return error

def ransac(matches, blobs1, blobs2):
    iterations = 1000
    N = len(matches)

    #Get all the non-empty feature matches
    features = list(np.where(matches!=-1)[0])
    
    #Reset max inliers counter
    maxInliers = 0
    bestAffineT = None
    inlinePts = [] 
    
    threshold = 1

    #Store all features and locations in inputs for 1st image and in bases for 2nd image
    for iter in range(iterations):
        bases = random.sample(features, 6)
        inputs = []
        
        for id in bases:
            inputs.append(int(matches[id]))
        
        coordX = []
        coordY = []
        coordT = []
       
        for i in range(6):          
            coordX.append(blobs1[bases[i]][0])
            coordY.append(blobs1[bases[i]][1])
            coordT.append(blobs2[inputs[i]][0])
            coordT.append(blobs2[inputs[i]][1])

        #Affine Transform for the current iteration
        B = np.array(coordT).T    
        
        A1 = np.array([[coordX[0], coordY[0], 0,0,1,0],[0,0,coordX[0], coordY[0],0,1]])
        A2 = np.array([[coordX[1], coordY[1], 0,0,1,0],[0,0,coordX[1], coordY[1],0,1]])
        A3 = np.array([[coordX[2], coordY[2], 0,0,1,0],[0,0,coordX[2], coordY[2],0,1]])
        A4 = np.array([[coordX[3], coordY[3], 0,0,1,0],[0,0,coordX[3], coordY[3],0,1]])
        A5 = np.array([[coordX[4], coordY[4], 0,0,1,0],[0,0,coordX[4], coordY[4],0,1]])
        A6 = np.array([[coordX[5], coordY[5], 0,0,1,0],[0,0,coordX[5], coordY[5],0,1]])
        A = np.concatenate((A1, A2, A3, A4, A5, A6))
        
        leastSquares = np.linalg.lstsq(A, B, rcond=None)

        lstqSoln = leastSquares[0]
        residual = leastSquares[1]
        rank = leastSquares[2]
        SingularCoefficent = leastSquares[3]

        M = np.array([[lstqSoln[0], lstqSoln[1]], [lstqSoln[2], lstqSoln[3]]])
        T = np.array([[lstqSoln[4]], lstqSoln[5]])
        
        #Find and add inliers
        numInline = 0
        inline = []
         
        for f in features:
            x, y = blobs1[f][0], blobs1[f][1]  
            xT, yT = blobs2[int(matches[f])][0], blobs2[int(matches[f])][1]
            xE = lstqSoln[0] * x + lstqSoln[1] * y + lstqSoln[4]
            yE = lstqSoln[2] * x + lstqSoln[3] * y + lstqSoln[5]
            error = calaculateError([xE, yE],[xT, yT])
            
            if error < threshold:
                numInline+=1
                inline.append(f)
        
        #Find the best model by inlier count and error values
        if numInline > maxInliers:
            maxInliers = numInline
            inlinePts = inline 
            bestAffineT = np.array([[lstqSoln[0],lstqSoln[1],lstqSoln[4]],[lstqSoln[2],lstqSoln[3],lstqSoln[5]]])
    
    print("Maximum Inliers: ", maxInliers)
    print( "Msize: " , len(features))
    print("Affine Transformation: ")
    print(bestAffineT)
    return inlinePts, bestAffineT   



