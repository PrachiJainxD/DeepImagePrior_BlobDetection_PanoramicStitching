# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2020
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 2

import matplotlib.pyplot as plt
from utils import imread
from depthFromStereo import depthFromStereo
import os

read_path = "../data/disparity/"

testExamples = ["cones", "teddy"]
for exampleIndex in range(len(testExamples)):
    im_name1 = "{}_im2.png".format(testExamples[exampleIndex])
    im_name2 = "{}_im6.png".format(testExamples[exampleIndex])

    #Read test images
    img1 = imread(os.path.join(read_path, im_name1))
    img2 = imread(os.path.join(read_path, im_name2))
    ws = 100
    #Compute depth
    depth = depthFromStereo(img1, img2, ws)

    #Show result
    plt.imshow(depth, cmap="jet")
    plt.show()
    save_path = "../../outputs/disparity/"
    save_file = im_name1+""
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    plt.savefig('{}_SSDws{}.png'.format(save_file, ws), cmap="jet")
