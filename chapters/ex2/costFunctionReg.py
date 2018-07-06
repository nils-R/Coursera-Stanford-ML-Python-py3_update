import numpy as np
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
    
    z = X.dot(theta) 
    J = -1/m * np.sum( y*np.log(sigmoid(z)) + ((np.ones(m)-y) * (np.log(np.ones(m) - sigmoid(z))) ) ) + (Lambda/(2*m)*np.sum(np.array(theta)[1:n]**2))
    
    error = sigmoid(X.dot(theta)) - y
        
    return J #, grad  <---- Python implementation is different from Matlab. See gradientFunctionReg instead

#     # ====================== YOUR CODE HERE ======================
#     # Instructions: Compute the cost of a particular choice of theta.
#     #               You should set J to the cost.
#     #               Compute the partial derivatives and set grad to the partial
#     #               derivatives of the cost w.r.t. each parameter in theta
#     # =============================================================

# =============================================================================

