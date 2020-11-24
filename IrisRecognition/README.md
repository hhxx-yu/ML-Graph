###IrisLocalization.py

This function is to localize the pupil. The function apply hough transform to detect both the circles of pupil and iris. Here we add restriction to pupil/iris into 120 * 120 centeral area of eye image to avoid misdetection.

##### Parameters:

image: the input image of the eye

##### Return:

iris: x y coordinates of center point of iris circle, and the radius

pupil: x y coordinates of center point of pupil circle, and the radius

### IrisNormalizaton.py

This function is to normalize iris area to a rectangular image of size 64 * 512.

##### Parameters:

image: eye image
pupil : pupil/inner circle center and radius
iris: iris/outer circle boundary center and radius

##### Return:

np.array(normalized): normalized iris image

```python
pupil_height, pupil_width, pupil_r = pupil
iris_height, iris_width, iris_r = iris

# Dectected circles for the pupil and the circle of an iris is usually not concentric
r = int(round(iris_r - np.hypot(pupil_height-iris_height, pupil_width-iris_width)))
if r < pupil_r + 15:
  r = 80 #set upper bound
maxh, maxw = image.shape
r2 = min(pupil_height, pupil_width, maxh-pupil_width, maxw-pupil_height)
r = min(r, r2)
max_r = r - pupil_r
  
# Normalize iris area to a rectangular image of size 64 by 512
M = 64
N = 512
normalized = []
for y in range(M):
  width_pix = []
  for x in range(N):
    theta = float(2*np.pi*x/N)
    hypotenuse = float(max_r*y/M) + pupil_r
    height = int(round((np.cos(theta) * hypotenuse) + pupil_height))
    width = int(round((np.sin(theta) * hypotenuse) + pupil_width))
    width_pix.append(image[width, height])
  normalized.append(width_pix)
```



### ImageEnhancement.py

This function divides the normalized image into 32 by 32 grids and equalizes the histogram of each grid. which give the enhance image

##### Input: 

image: normalized iris area

##### Return:

enhance: enhanced iris area

```python
enhanced = image.copy()
block = 32
for i in range(2):
  for j in range(16):
    # Define each 32 by 32 grid iteratively
    height = i*block
    width = j*block
    grid = enhanced[height:height+block, width:width+block]
    enhanced[height:height+block, width:width+block] = cv2.equalizeHist(grid)
```

### FeatureExtraction.py

In this function, we set region of interest (ROI) is a matrix with shape 48*512 and apply gabor filter to extract iris feature.

##### Input: 

image: enhanced normalized iris area

##### Return:

feature: iris feature (each feature has the dimension of 1,536 (48 * 512 )/ (8 * 8) *2 *2)

#### def gabor_filter:

This function implements the Gabor filter function which is described in Li Ma's paper.
```python
m1 = np.cos(2*np.pi*f*np.sqrt(x**2+y**2)) 
filtered = 1/(2*np.pi*space_constant_x*space_constant_y)
   *np.exp(-1/2*((x**2)/(space_constant_x**2)+(y**2)/(space_constant_y**2)))*m1
```

#### def FeatureExtraction:

we use convolution method to convolve two filters on the roi to get two filtered images
```python
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
```

we calculate the mean and mean absolute deviation of every 8*8 blocks of roi. This is a method proposed by Li Ma. 
```python
for i in range(int(height/block)): #block=8
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
```


### IrisMatching.py

#### def fisher_LD:
It transforms the feature vector from FeatureExtraction to a vector in a low-dimensional feature space.
```python
:param train_img: feature vector of training images 
:param train_label: labels of training images 
:param img: feature vector of images to be transformed
:param n_components: dimension of the new transformed feature vector
:return: transformed feature vecter
sklearn_lda = LDA(n_components = n_components)
sklearn_lda.fit(train_img, train_label)
img_t=sklearn_lda.transform(img)
```

#### def LLE:
It also transforms the feature vector and it is used when n components is over nclass-1 as a supplement for fisher_LD
```python
embedding = LocallyLinearEmbedding(n_neighbors=201,n_components=n_components)
embedding.fit(train_img, train_label)
img_t=embedding.transform(img)
return img_t
```

#### def classification
It trains the nearest centroid classification with the transformed vectors from fisher_LD and LLE, outputing the predicting label, which outputs the predicting labels of testing images and distance of all test feature vectors to the centroids
```
:param train_img: feature vector of training images 
:param train_label: labels of training images 
:param test_img: feature vector of test images
:param distance: 'l1','l2' or 'cosine' 
"""
clf = NearestCentroid(metric=distance)
clf.fit(train_img,train_label)
predict_label=clf.predict(test_img)
dist = pairwise_distances(test_img, clf.centroids_, metric=clf.metric)
```

### PerformanceEvaluation.py

#### evaluation:
It calculates the CRR of the model based on the result of classification.
```python
CRR=sum(predict_label==test_label)/len(test_label)
```

#### label2matrix:
A fuction transforms the label to a 0-1 matrix, which is used to simplify the calculation.
```python
label=list(map(int,label))
label = np.array(label)
uq_la = np.unique(label)
c = uq_la.shape[0]
n = label.shape[0]
label_mat = np.zeros((n,c))
for i in range(c):
    index = (label == i+1)
    label_mat[index,i]=1
```

#### FMRvsFNMR
It calculates the false match rate and false non-match rate based on the distance from classification and different thresholds.
```python
 """
:param test_dist: distance of the test feature vectors to the centroids
:param test_label: labels of test images
:param threshold_range: a set of thresholds
FNMR=[]
FMR=[]
for threshold in threshold_range:
    match=(test_dist<threshold)
    labelmat=label2matrix(test_label)
    total_Positive=sum(sum(labelmat==1))
    total_Negative=sum(sum(labelmat==0))
    False_Positive=total_Positive-sum(sum((labelmat*match)==1))
    False_Negative=total_Negative-sum(sum((abs(labelmat-1)*abs(match-1))==1))
    FNMR.append(False_Positive/total_Positive)
    FMR.append(False_Negative/total_Negative)
```

### IrisRecognition.py
The main function which outputs the correct recognition rate and a plot of ROC-curve.

#### Import
```python
import os
import cv2
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestCentroid 
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
from IrisMatching import fisher_LD, classification, LLE
from PerformanceEvaluation import evaluation, FMRvsFNMR
```

#### Read in the data
Read in the images and assign them to the relative train data, train label, test data and test label.
```python
data_dir = os.getcwd() + '/CASIA Iris Image Database (version 1.0)'
classes = os.listdir(data_dir)
if ".DS_Store" in classes:
    classes.remove(".DS_Store")
traindata = []
trainlabel=[]
testdata = []
testlabel=[]
train_dir='/1/'
test_dir='/2/'
for cls in classes:
    files = os.listdir(data_dir+"/"+cls+train_dir)
    if "Thumbs.db" in files:
        files.remove("Thumbs.db")
    for f in files:        
        img = cv2.imread(data_dir+"/"+cls+"/"+train_dir+f)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        traindata.append(img)
        trainlabel.append(cls)
    files_test = os.listdir(data_dir+"/"+cls+test_dir)
    if "Thumbs.db" in files_test:
        files_test.remove("Thumbs.db")
    for f in files_test:        
        img = cv2.imread(data_dir+"/"+cls+"/"+test_dir+f)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        testdata.append(img)
        testlabel.append(cls)
```

#### Localization, Normalization, Enhancement and FeatureExtraction
Perform localization, normalization, enhancement and featureExtraction on the images in both training and testing set.
```python
traindata_feature=[]
for i in range(len(traindata)):
    print(i)
    r_outter,r_inner=IrisLocalization(traindata[i])
    img_norm=IrisNormalization(traindata[i], r_inner,r_outter)
    img_enhance=ImageEnhancement(img_norm)
    img_FE=FeatureExtraction(img_enhance)
    traindata_feature.append(img_FE)

testdata_feature=[]
for i in range(len(testdata)):
    print(i)
    r_outter,r_inner=IrisLocalization(testdata[i])
    img_norm=IrisNormalization(testdata[i], r_inner,r_outter)
    img_enhance=ImageEnhancement(img_norm)
    img_FE=FeatureExtraction(img_enhance)
    testdata_feature.append(img_FE)
```
#### Dimension Reduction and Classification
Perform fisher linear discriminant and locally linear embedding on the dataset. And then, using the nearest center classifier based on both reduced and full feature vectors. It outputs table3 for every selection of the length of the reduced feature vectors.
```python
acc=[]
measure=('l1','l2','cosine')
r=range(1,302,20)
for n_components in r:
    if n_components<107:
        traindata_feature_LDA=fisher_LD(traindata_feature, trainlabel, traindata_feature, n_components)
        testdata_feature_LDA=fisher_LD(traindata_feature, trainlabel, testdata_feature, n_components)
    else:
        traindata_feature_LDA=LLE(traindata_feature, trainlabel, traindata_feature, n_components)
        testdata_feature_LDA=LLE(traindata_feature, trainlabel, testdata_feature, n_components)
    test_predict_label_LDA=[]
    test_dist_LDA=[]
    test_CRR_LDA=[]
    for distance in measure:
        test_predict_label, test_dist=classification(traindata_feature_LDA, trainlabel, testdata_feature_LDA, distance)
        test_predict_label_LDA.append(test_predict_label)
        test_dist_LDA.append(test_dist)
        test_CRR_LDA.append(evaluation(test_predict_label, testlabel))

    test_predict_label_full=[]
    test_dist_full=[]
    test_CRR_full=[]
    for distance in measure:
        test_predict_label, test_dist=classification(traindata_feature, trainlabel, testdata_feature, distance)
        test_predict_label_full.append(test_predict_label)
        test_dist_full.append(test_dist)
        test_CRR_full.append(evaluation(test_predict_label, testlabel))
    acc.append(test_CRR_LDA)
    tb = pt.PrettyTable()
    tb.field_names = ["Measure", "CRR_full", "CRR_LDA"]
    for i in range(3):
        tb.add_row([measure[i],format(test_CRR_full[i],'.3f'), format(test_CRR_LDA[i],'.3f')])   
    print(tb)
```

#### Results and Plots
In addition to the CRR table for the 3 measures above, this part outputs the ROC curve based on FMR and FNMR, a table of FMR and FNMR based on different threshold and a plot of CRR based on different choice of length of reduced feature vectors.
```python
range_list= (0.775,0.8,0.825,0.85,0.875,0.9)
FMR, FNMR=FMRvsFNMR(test_dist_LDA[2], testlabel, range_list)
plt.plot(FMR,FNMR)
plt.xlabel("FMR")
plt.ylabel("FNMR")
plt.show()
tb2 = pt.PrettyTable()
tb2.field_names = ["Threshold", "False Match Rate", "False Non-Match Rate"]
for i in range(len(range_list)):
    tb2.add_row([range_list[i], format(FMR[i],'.3f'),format(FNMR[i],'.3f')])
print(tb2)
for i in range(3):
    plt.plot(r,np.array(acc)[:,i])
    plt.legend(measure)
    plt.xlabel("n_components")
    plt.ylabel("CRR")
plt.show()
```
##### Correct Recognition Rate
|Measure | CRR_full| CRR_LDA|
|:--|:--|:--|
|l1 | 0.611 |  0.762 |
|l2 | 0.616 |  0.771 |
|cosine | 0.653|  0.808 |

##### False Match Rate and False Non-Match Rate
|Threshold | FMR| FNMR|
|:--|:--|:--|
|0.775 | 0.002 |  0.227 |
|0.8 | 0.004 |  0.197 |
|0.825 | 0.009|  0.160 |
|0.85 | 0.019|  0.125 |
|0.875 | 0.036|  0.109 |
|0.9 | 0.065|  0.076 |


### Limitation
The localization of iris is not very accurate, and this may cause the error for the matching step, weakening the CRR performance. Also,Bootstrap is not used for verification, which may lead to a less trustful select of threshold.

###Refrence

1. Personal Identification Based on Iris Texture Analysis IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 25, NO. 12, DECEMBER 2003 https://pdfs.semanticscholar.org/e709/fd0f125fdd78769659f5e46c05482331ce54.pdf
2. CASIA Iris Image Database Version 1.0 - Biometrics Ideal Test