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
#read in the images
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

# Localization, Normalization, Enhancement and FeatureExtraction
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

#Fisher linear discriminant and nearest center classifier
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
# Output the result
    tb = pt.PrettyTable()
    tb.field_names = ["Measure", "CRR_full", "CRR_LDA"]
    for i in range(3):
        tb.add_row([measure[i],format(test_CRR_full[i],'.3f'), format(test_CRR_LDA[i],'.3f')])   
    print(tb)
#Plot for ROC
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
#Plot for CRR towards different feature space
for i in range(3):
    plt.plot(r,np.array(acc)[:,i])
    plt.legend(measure)
    plt.xlabel("n_components")
    plt.ylabel("CRR")
plt.show()
    