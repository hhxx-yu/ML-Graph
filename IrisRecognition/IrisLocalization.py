import cv2
import numpy as np

def getimagecenter(image):
    """
    Localize image center

    :param image: image
    :return: center point height width
    """
	height, width = image.shape[:2]
	return height/2, width/2

def getcenterregion(image, x, y):
    """
    Localize centeral region of image

    :param image: image
    :param x: center point height
    :param y: center point width
    :return: 120*120 area around center 
    """    
    x=int(x)
    y=int(y)
    res = image[x-60:x+60, y-60:y+60]
    return res

def getimagecentroid(image):
    """
    Localize centeral based on contour detection

    :param image: center part of image
    :return: centroid 
    """   
    ret,thresh_img = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print (cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    M = None
    max_m00 = 0.0
    for i in contours:
        cnt = i
        temp = cv2.moments(cnt)
        if temp['m00'] > max_m00:
            M = temp
            max_m00 = temp['m00']
    return int(M['m01']/M['m00']), int(M['m10']/M['m00'])
    


def IrisLocalization(image):
    """
    Localize pupil (inner circle) and iris (outer circle)

    :param image: eye image
    :return: iris pupil coordinate  
    """      
    x, y = getimagecenter(image)
    height, width = image.shape

    temp = getcenterregion(image, x, y)
    x_1, y_1 = getimagecentroid(temp)
    x = x-60+x_1
    y = y-60+y_1
    x=int(x)
    y=int(y)

    #restrict center area to pupil and iris, apply gaussianblur and houghcircle to localize pupil and iris

    temp = image[max(x-120,0):min(x+120, height), max(y-120,0):min(y+120,width)]
    temp = cv2.GaussianBlur(temp,(3,3),0)
    circles_tmp = cv2.HoughCircles(temp,cv2.HOUGH_GRADIENT, dp=1,minDist=200,param1=100,param2=10,minRadius=70,maxRadius=120)
    iris = circles_tmp[0,:,:]
    iris = np.uint16(np.around(iris)) 
    iris[0][0] += max(y-120,0)
    iris[0][1] += max(x-120,0)


    temp = image[max(x-60,0):min(x+60, height), max(y-60,0):min(y+60,width)]
    temp = cv2.GaussianBlur(temp,(3,3),0) 
    circles_tmp = cv2.HoughCircles(temp,cv2.HOUGH_GRADIENT, dp=1,minDist=200,param1=100,param2=10,minRadius=20,maxRadius=55)
    pupil = circles_tmp[0,:,:] 
    pupil = np.uint16(np.around(pupil))
    pupil[0][0] += max(y-60,0)
    pupil[0][1] += max(x-60,0)
    

    return iris[0], pupil[0]