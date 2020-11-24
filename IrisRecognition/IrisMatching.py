from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestCentroid 
from sklearn.manifold import LocallyLinearEmbedding

def fisher_LD(train_img, train_label, img, n_components):
    """
    It transforms the feature vector to one in a low-dimensional feature space.
    
    :param train_img: feature vector of training images 
    :param train_label: labels of training images 
    :param img: feature vector of images to be transformed
    :param n_components: dimension of the new transformed feature vector
    :return: transformed feature vecter 
    """
    sklearn_lda = LDA(n_components = n_components)
    sklearn_lda.fit(train_img, train_label)
    img_t=sklearn_lda.transform(img)
    return img_t

def LLE(train_img, train_label, img, n_components):
    """
    It transforms the feature vector to one in a low-dimensional feature space.
    
    :param train_img: feature vector of training images 
    :param train_label: labels of training images 
    :param img: feature vector of images to be transformed
    :param n_components: dimension of the new transformed feature vector
    :return: transformed feature vecter 
    """
    embedding = LocallyLinearEmbedding(n_neighbors=201,n_components=n_components)
    embedding.fit(train_img, train_label)
    img_t=embedding.transform(img)
    return img_t
    
def classification(train_img, train_label, test_img, distance):
    """
    It trains the nearest centroid classification and output the predicting label.
    
    :param train_img: feature vector of training images 
    :param train_label: labels of training images 
    :param test_img: feature vector of test images
    :param distance: 'l1','l2' or 'cosine'
    :return: predicting labels of testing images and distance of all test feature vectors to the centroids
    """
    clf = NearestCentroid(metric=distance)
    clf.fit(train_img,train_label)
    predict_label=clf.predict(test_img)
    dist = pairwise_distances(test_img, clf.centroids_, metric=clf.metric)
    return predict_label, dist