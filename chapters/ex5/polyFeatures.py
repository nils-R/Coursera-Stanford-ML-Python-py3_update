import numpy as np


def polyFeatures(X, p):
    """takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    """
    # You need to return the following variables correctly.
    X_poly = np.zeros((X.size, p))
    
    for p in range(0,p+1):
        X_poly[:,p-1] = np.power(X,p)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Given a vector X, return a matrix X_poly where the p-th
    #               column of X contains the values of X to the p-th power.
    #
    # =========================================================================

    return X_poly
