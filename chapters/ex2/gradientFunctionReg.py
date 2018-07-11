import numpy as np
from sigmoid import sigmoid


def gradientFunctionReg(theta, X, y, Lambda, vectorized=True):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Copy from previous function:
      
    m = len(y)   # number of training examples
    n = len(theta)
    
    z = X.dot(theta) 
    error = sigmoid(z) - y
    grad = np.zeros(n)
    
    if vectorized == True:
        grad = 1/m* (error.dot(X) + Lambda * np.concatenate((np.zeros(1),theta[1:])) )
    else:
        for j in range(n):
            if j == 0:
                grad[j] = 1/m*np.sum(error.dot(X[:,j]))
            else:
                grad[j] = 1/m*np.sum(error.dot(X[:,j])) + Lambda/m*theta[j]

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    # =============================================================

    return grad

