import numpy as np
from sigmoid import sigmoid
import sys
if '../ex2/' not in sys.path:
    sys.path.append('../ex2/')

def lrCostFunction(theta, X, y, Lambda=0):
    """ computes the cost of using
        theta as the parameter for regularized logistic regression and the
        gradient of the cost w.r.t. to the parameters.
    """
   
    m = len(y)   # number of training examples
    n = len(theta)

    z = X.dot(theta)
    cost1 = np.log(sigmoid(z))
    cost0 = np.log(1 - sigmoid(z))
    regularization = Lambda/(2*m)*np.sum(np.array(theta)[1:n]**2)
    
# =============================================================================
#     print(cost1.shape, cost0.shape)
#     print(zeros.shape)
#     print(theta[1:].shape)
#     print(np.concatenate((zeros,theta[1:])).shape)
#     print( np.multiply(y, cost1).shape)
#     print( np.multiply(y, cost0).shape)
# =============================================================================
    
    J = -1/m * np.sum( np.multiply(y, cost1) + np.multiply(1.0-y, cost0) ) + regularization 
    
    #J = -1/m * np.sum( y * np.log(sigmoid(z)) + ((np.ones(m)-y) * (np.log(np.ones(m) - sigmoid(z))) ) ) + (Lambda/(2*m)*np.sum(np.array(theta)[1:n]**2))
    #error = sigmoid(z) - y
    #grad = 1/m* (X.T.dot(error) + Lambda * np.concatenate((zeros,theta[1:])) )
    
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

    return J