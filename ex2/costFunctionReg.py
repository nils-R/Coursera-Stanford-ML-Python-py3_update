import numpy as np
from costFunction import costFunction
from sigmoid import sigmoid


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples
    n = len(theta)

    theta = np.matrix(theta).T
    X = np.matrix(X)
    y = np.matrix(y).T
    
    z = X.dot(theta)
    J = -1/m * ( np.sum( y*np.log(sigmoid(z)) + (np.ones(m)-y) * np.log(np.ones(m) - sigmoid(z)) ) ) + (Lambda/(2*m)*np.sum(np.array(theta)[1:n]**2))
    
    error = sigmoid(X * theta) - y
    grad = np.zeros(n)
    

    
    grad = 1/m*(X.T.dot(error)+np.sum(np.array(theta)[1:n]))
        
    return J, grad
#     # ====================== YOUR CODE HERE ======================
#     # Instructions: Compute the cost of a particular choice of theta.
#     #               You should set J to the cost.
#     #               Compute the partial derivatives and set grad to the partial
#     #               derivatives of the cost w.r.t. each parameter in theta
#     # =============================================================

# =============================================================================


def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg