import numpy as np
from sigmoid import sigmoid

def gradientFunction(theta, X, y, vectorized=True):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    m = len(y)   # number of training examples
    
    if X.shape[1]:
        n = X.shape[1]
    else:
        n = 1

    theta = np.matrix(theta).T
    X = np.matrix(X)
    y = np.matrix(y).T
        
    error = sigmoid(X * theta) - y
    grad = np.zeros(n)
    
    if vectorized == True:
        grad = 1/m*X.T.dot(error)
    else:
        for i in range(n):
            grad[i] = 1/m*np.sum((X[:,i]).T.dot(error))
    
            
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    # =============================================================

    return grad
