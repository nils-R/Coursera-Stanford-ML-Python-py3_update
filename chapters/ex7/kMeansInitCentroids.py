import numpy as np


def kMeansInitCentroids(X, K):
    """returns K initial centroids to be
    used with the K-Means on the dataset X
    """

    # You should return this values correctly
    centroids = X[np.random.choice(X.shape[0], K, replace=False), :]
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    #
    # =============================================================

    return centroids
