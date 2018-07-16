import numpy as np


def findClosestCentroids(X, centroids):
    """returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

    # Set K
    K = len(centroids)
    
    # Determine number of training examples
    m = X.shape[0]
    
    # You need to return the following variables correctly.
    idx = np.zeros(m, dtype=int)
    idk = np.zeros(K, dtype=float)
    
    for i in range(m):
        for k in range(K):
            idk[k] = np.linalg.norm(X[i]-centroids[k])
        idx[i] = idk.argmin()

        #for k in range(K):
        #        idk[k] = np.linalg.norm(X[i,:]-centroids[k,:])
        #np.linalg.norm(X[i,:]-centroids[k,:])
        
        
    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the
    #               range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
    #
    # =============================================================

    return idx

def test_centroids():
    X_t = np.sin([range(1,51)]).reshape(5,10).T
    cent = X_t[6:10,:]
    return findClosestCentroids(X_t, cent) + 1 # plus one to compare with octave solution