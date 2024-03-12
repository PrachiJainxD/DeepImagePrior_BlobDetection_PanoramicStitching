#imported libraries
import numpy as np

def depthFromStereo(img1, img2, ws):
    #We take mean of all three channels of the first and second images to get a grayscale approximation of the images
    img1 = (img1[:,:,0]+img1[:,:,1]+img1[:,:,2])/3
    img2 = (img2[:,:,0]+img2[:,:,1]+img2[:,:,2])/3  

    height1, width1 = np.shape(img1)
    height2, width2 = np.shape(img2)
    
    #Initialize depth mean to all zeros
    depth = np.zeros((height1,width1))
    difference = set()
    
    padSize = ws//2
    padHeight, padWidth = ws//2, ws//2
    
    #Handling even window sizes
    if ws % 2 ==0: 
        padWidth -= 1
        
    pad1 = np.pad(img1, ((padSize, padSize), (padSize, padSize)), mode='constant', constant_values=(0,0))
    pad2 = np.pad(img2, ((padSize, padSize), (padSize, padSize)), mode='constant', constant_values=(0,0))
    
    metric = 0
    
    if metric == 0:
    #SSD
        #Traverse through all Epipolar Lines
        for i in range(height1):
            pX = i + padSize
            patch = [np.zeros((ws, ws))] * width2
            #For each pixel on epipolar lines
            for j in range(width2):
                pY = j + padSize
                #Patch Box
                patch[j] = pad2[pX - padHeight:pX+padHeight, pY-padWidth:pY + padWidth]
            
            for k in range(width1):
                x, dX = k, k
                pv = k + padSize
                patch1 = pad1[pX - padHeight:pX+ padHeight, pv-padWidth:pv + padWidth]
                minScore = float('inf')
                #Match the image patches on an epipolar line
                for j in range(k):
                    patch2 = patch[j]
                    SSD = float('inf')
                    SSD = np.sum(np.multiply(patch1 - patch2, patch1 - patch2))
                    if SSD <= minScore:
                        dX = j
                        minScore = SSD
                try:
                    disparity = abs(x - dX)
                    depth[i,k] = min(1/disparity, 1)
                    difference.add(disparity)
                except:
                    depth[i,k] = 1
                    difference.add(0)                    
    elif metric == 1:
    #Normalized correlation 
        for i in range(height1):
            pX = i + padSize 
            patch = [np.zeros((ws,ws))]* width2
            for j in range(width2):
                pY = j + padSize
                patch[j] = pad2[pX-padHeight:pX+padHeight, pY-padWidth:pY+padWidth]

            for k in range(width1):
                x, dX = k, k
                pv = k + padSize
                patch1 = pad1[pX-padHeight:pX+padHeight, pv-padWidth:pv+padWidth]
                
                maxScore = -float('inf')
                for j in range(width2):
                    patch2 = patch[j]
                    norm_corr = - float('inf')
                    norm_corr = (1 / width1) * (np.sum(patch1 - np.mean(patch2)) / np.sqrt(np.var(patch1) * np.var(patch2)))               
                    if norm_corr > maxScore:
                        dX = j
                        maxScore = norm_corr
                try:
                    disparity = abs(x - dX)
                    depth[i,k] = min(1/disparity, 1)
                    difference.add(disparity)
                except:
                    depth[i,k] = 1
                    difference.add(0)        
                    
    return depth