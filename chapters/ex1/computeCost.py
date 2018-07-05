import numpy as np


def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    inner = np.power(((X.dot(theta.T)) - y), 2)
    J = np.sum(inner) / (2 * m)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.


# =========================================================================

    return J


