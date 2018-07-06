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

    z = X.dot(theta)        
    error = sigmoid(z) - y
    grad = np.zeros(n)
    
    if vectorized == True:
        grad = 1/m*error.dot(X)
    else:
        for j in range(n):
            grad[j] = 1/m*np.sum(error.dot(X[:,j]))
    
            
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    # =============================================================

    return grad
