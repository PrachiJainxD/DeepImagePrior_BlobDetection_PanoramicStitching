**1. Depth from disparity**
   - Denoising images with
     1. AutoEncoder with Batch Normalization and Bilinear Upsampling
     2. Deeper Architectures, those with Skip Connections, as well as Averaging the Results across iterations.
    
**2. Scale-space blob detection**
   -  Built a Laplacian scale space, starting with some initial scale and going for n iterations:
   -  (a) Filter image with scale-normalized Laplacian at current scale.
   -  (b) Save the square of Laplacian response for current level of scale space.
   -  (c) Increase scale by a factor k.
   -   Perform non-maximum suppression in scale space.
   -   Display resulting circles at their characteristic scales for points above a threshold.

**3. Image stitching**
   - Implemented the RANSAC algorithm to stitch two images. The input to the algorithm are two images which are related by an unknown transformation.
   - Used the blobs detector implemented to extract keypoints and extract feature descriptors on them.
   - Then estimated an affine transformation using feature matching and RANSAC to produce a combined image.
