import numpy as np
from gradientFunction import gradientFunction
from sigmoid import sigmoid


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Copy from previous function:
    
    m = len(y)   # number of training examples
    n = len(theta)
    
    z = X.dot(theta) 

    error = sigmoid(X.dot(theta)) - y
    grad = np.zeros(n) 
    grad = 1/m*(X.T.dot(error)+np.sum(np.array(theta)[1:n]))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    # =============================================================

    return grad