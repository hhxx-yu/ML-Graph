import numpy as np


def gabor_filter(x, y, space_constant_x, space_constant_y, f):
    """
    Define gabor filters to get details of the iris

    :param x: x-coordinate
    :param y: y-coordinate
    :param space_constant_x: space constants of the Gaussian envelope along the x axis
    :param space_constant_y: space constants of the Gaussian envelope along the y axis
    :param f: frequency of the sinusoidal function
    :return: filtered image
    """
    m1 = np.cos(2*np.pi*f*np.sqrt(x**2+y**2)) 
    filtered = 1/(2*np.pi*space_constant_x*space_constant_y)*np.exp(-1/2*((x**2)/(space_constant_x**2)+(y**2)/(space_constant_y**2)))*m1

    return filtered


def FeatureExtraction(image, block=8):
    """
    Converts image to a feature

    :param image: enhanced normalized image
    :param block: filter block size
    :return: feature 
    """


    roi = np.array(image[:48, :512]) #upper portion of a normalized iris image 48*512
    height, width = roi.shape
    filtered1 = np.empty((height, width))
    filtered2 = np.empty((height, width))
    side = int((block-1)/2)
    feature = []
    space_constant_x1=3
    space_constant_x2=4.5
    space_constant_y=1.5

    for i in range(height):
        for j in range(width):

            cell1 = 0
            cell2 = 0
            for m in range(i-side, i+side+1):
                for n in range(j-side, j+side+1):

                    if (0 < m < height) and (0 < n < width):
                        cell1 += roi[m, n]*gabor_filter(i-m, j-n, space_constant_x1, space_constant_y, 1/1.5)
                        cell2 += roi[m, n]*gabor_filter(i-m, j-n, space_constant_x2, space_constant_y, 1/1.5)
            filtered1[i, j] = cell1
            filtered2[i, j] = cell2

    for i in range(int(height/block)):
        for j in range(int(width/block)):

            height1 = i*block
            height2 = height1+block
            width1 = j*block
            width2 = width1+block
            grid1 = filtered1[height1:height2, width1:width2]
            grid2 = filtered2[height1:height2, width1:width2]

            #filter1
            absolute = np.absolute(grid1)
            mean = np.mean(absolute)
            feature.append(mean)
            std = np.mean(np.absolute(absolute-mean))
            feature.append(std)

            # filter 2
            absolute = np.absolute(grid2)
            mean = np.mean(absolute)
            feature.append(mean)
            std = np.mean(np.absolute(absolute-mean))
            feature.append(std)

    return feature


#from FeatureExtraction import FeatureExtraction