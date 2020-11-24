# image_segmentation 
###
upload 2 .py document for 2 pics.

### Work Flow
1. Read image data and turn to rgb color space. 
2. Use k-means algorithm to classify color to different category, for distance simply normalize difference by standard derivation to get better clustering result.
3. Use opencv threshold to get countours, find aimed part (face) in contours, then draw rectangle and output the face segmentation result with bounding box surrounding the target area.

### Variables explain
| variables | meaning |
| --- | --- |
| k | how many color categories |
| imgPath | the path of the image folder |
| imgFilename | the name of the image file |
| savedImgPath | the path of the folder you will save the image |
| savedImgFilename | the name of the output image                   |
| image_rgb | data in RGB color space                        |
| center           | cluster center during k-means                  |
| result           | labeled data after k-means clustering          |
| threshed_img     | threshed image by label                        |
| contours         | contours part of image in threshed image       |

### Function explain

##### dist
This function calculate distance between center and aim point in RGB color space, here we simply normalize difference by standard derivation to get better clustering result.

input: 

output: 
    
### Result
![face_d2_face](/Users/a/Downloads/face_d2_face.jpg)



### Limitations
Acutually I have tried to use k-means in opencv cv.kmeans to cluster the image data( both RGB and LAB space):

```python
# reshape the image to a 2D array of pixels and 3 color values (RGB/LAB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)
# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# number of clusters (K)
k = 6
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# convert back to 8 bit values
centers = np.uint8(centers)
# flatten the labels array
labels = labels.flatten()
result = labels.reshape(image.shape)
_, threshed_img = cv2.threshold(result, 4, 5, cv2.THRESH_BINARY)
threshed_img = cv2.convertScaleAbs(threshed_img)
contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

However, the clustering result is not that satisfying,  it sometimes shows more than one contour (more than one rectangle display in picture ) .

My script detecting multi-faces simply changes the parameter to choose cluster and contours. Color based segmentation via K-means helps , but face detecting should consider more features.