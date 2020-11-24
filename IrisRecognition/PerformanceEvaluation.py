import numpy as np

def evaluation(predict_label, test_label):
    """
    It calculates the CRR of the model
    :param predict_label: predicting labels of testing images
    :param test_label: labels of testing images
    :return: correct recognition rate
    """
    CRR=sum(predict_label==test_label)/len(test_label)
    return CRR

def label2matrix(label):
    """
    It transforms the label to a 0-1 matrix
    :param label: labels of images
    :return: 0-1 matrix which indicates the labels of the images
    """
    label=list(map(int,label))
    label = np.array(label)
    uq_la = np.unique(label)
    c = uq_la.shape[0]
    n = label.shape[0]
    label_mat = np.zeros((n,c))
    for i in range(c):
        index = (label == i+1)
        label_mat[index,i]=1
    return label_mat

def FMRvsFNMR(test_dist, test_label, threshold_range):
     """
     It calculates the false match rate and false non-match rate
     :param test_dist: distance of the test feature vectors to the centroids
     :param test_label: labels of test images
     :param threshold_range: a set of thresholds
     :return: false match rate and false non-match rate
     """
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
     return FMR, FNMR