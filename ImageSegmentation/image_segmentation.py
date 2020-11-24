import cv2
import copy
import os
import numpy as np
import matplotlib.pyplot as plt

def dist_color(center, point, sigma):
    return float(np.sqrt(((center[0] - point[0]) / sigma[0]) ** 2 + ((center[1] - point[1]) / sigma[1]) ** 2 + (
                (center[2] - point[2]) / sigma[2]) ** 2))

def image_seg(imgPath, imgFilename, savedImgPath, savedImgFilename, k):
# """
#     parameters:
#     imgPath: the path of the image folder. Please use relative path
#     imgFilename: the name of the image file
#     savedImgPath: the path of the folder you will save the image
#     savedImgFilename: the name of the output image
#     k: the number of clusters of the k-means function 
#     function: using k-means to segment the image and save the result to an image with a bounding box
# """
    #Write your k-means function here

    # Read Image
    imagefile = os.path.join(imgPath, imgFilename)
    image1 = cv2.imread(imagefile)
    img_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
     # image_rgb.shape
    # m,n = image_rgb.shape[0:2]

    # Initialize centers
    np.random.seed(100)
    center_m = np.random.randint(0, img_rgb.shape[0], size=k)
    center_n = np.random.randint(0, img_rgb.shape[1], size=k)
    centers = np.zeros((k, 3))

    for i in range(0, k):
        p = img_rgb[center_m[i]][center_n[i]].reshape(1, 3)
        centers[i] = p

    #print(centers)
    center_old = np.zeros(centers.shape)
    result = np.zeros(img_rgb[:, :, 0].shape)
    #prepare for dist calculation
    sigma = np.array([np.std(img_rgb[:, :, 0]), np.std(img_rgb[:, :, 1]), np.std(img_rgb[:, :, 2])])
    error = 1
    #update
    while error != 0:
        for i in range(0, img_rgb.shape[0]):
            for j in range(0, img_rgb.shape[1]):
                tmp = []
                for z in range(0, k):
                    tmp.append(dist_color(centers[z], img_rgb[i][j], sigma))

                result[i][j] = np.argmin(tmp) #minimize cost function
        center_old = copy.deepcopy(centers)

        for z in range(0, k): #update center
            temp = np.zeros((1, 3))
            for i in range(0, img_rgb.shape[0]):
                for j in range(0, img_rgb.shape[1]):
                    if result[i][j] == z:
                        temp = np.vstack((temp, img_rgb[i][j]))

            temp = np.delete(temp, (0), axis=0)
            centers[z] = np.mean(temp, axis=0)

            error = 0

            for y in range(0, k):
                error = error + dist_color(centers[y], center_old[y], sigma)

        result = result.astype(np.uint8)# convert back to 8 bit values
    # threshold image
    _, threshed_img = cv2.threshold(result, 3, 5, cv2.THRESH_BINARY)
    # plt.figure()
    # plt.imshow(threshed_img, cmap="gray")
    # plt.show() test threshold
    threshed_img = cv2.convertScaleAbs(threshed_img)
    contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # print(x,y,w,h,np.mean(image_rgb[y:y + h][x:x + w]))#<- find best parameter
        if np.mean(img_rgb[y:y + h][x:x + w]) > 120 and w > 20:
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
 
    
    #dislay
    plt.figure()
    plt.imshow(img_rgb, cmap="gray")
    plt.show()

    #save pic

    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(savedImgPath, savedImgFilename), img_rgb)
    cv2.waitKey(0)

if __name__ == "__main__":
    imgPath = "../hw1"
    imgFilename = "face_d2.jpg"
    savedImgPath = r'../hw1'
    savedImgFilename = "face_d2_face.jpg"
    k = 6
    hy2635_HuiqianYu_kmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k)