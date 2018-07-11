import numpy as np


def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    
    #try:
    #    X.shape[1]
    #except:
        #X = X[:,None]
        #theta = theta[:,None]
        #y = y[:,None]
        
    #X = np.column_stack((np.ones(m), X))
    if theta.size == 1:
        np.concatenate((np.ones(1),theta))
    
    hx = X.dot(theta)
    cost = np.sum((hx-y)**2)
    regularization = Lambda*np.sum(theta[1:]**2)
    
    if y.size == 0:
        J = 0
        grad = 0
    else:
        J = 1/(2*m) * (cost + regularization)   
        grad = 1/m * ( X.T.dot(hx-y) + Lambda * np.concatenate((np.zeros(1),theta[1:])) )

    # ====================== YOUR CODE HERE ===================================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #
    # =========================================================================

    return J, grad

