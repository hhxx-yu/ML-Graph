import cv2

def ImageEnhancement(image):
    """
    It divides the normalized image into 32 by 32 grids and equalizes the histogram of each grid.

    :param image: normalized iris image
    :return: enhance image by histogram equalization
    """

    enhanced = image.copy()
    block = 32
    for i in range(2):
        for j in range(16):
            # Define each 32 by 32 grid iteratively
            height = i*block
            width = j*block
            grid = enhanced[height:height+block, width:width+block]
            enhanced[height:height+block, width:width+block] = cv2.equalizeHist(grid)

    return enhanced

#from ImageEnhancement import ImageEnhancement