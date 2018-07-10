import numpy as np
from sigmoid import sigmoid
import sys
if '../ex2/' not in sys.path:
    sys.path.append('../ex2/')

def lrGradient(theta, X, y, Lambda=0.1):
    """ computes the cost of using
        theta as the parameter for regularized logistic regression and the
        gradient of the cost w.r.t. to the parameters.
    """
   
    m = len(y)   # number of training examples
    n = len(theta)

    #zeros = np.zeros((1,1))
    zeros = np.zeros(1)
    
    z = X.dot(theta)

    error = sigmoid(z) - y
    
    #print(X.T.shape, error.shape)
    #print(X.T.dot(error).shape)
    #print(np.concatenate((zeros,theta[1:])).shape)
    
    grad = 1/m* (X.T.dot(error) + Lambda * np.concatenate((zeros,theta[1:])) )
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X * theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.
    #
    #  =============================================================

    return grad