from computeCostMulti import computeCostMulti
import numpy as np

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples
    theta_temp = np.zeros(len(theta))
    
    for i in range(num_iters):
        
        e = (X.dot(theta.T) - y)
        
        for j, theta_j in enumerate(theta):
            j_deriv = e*X[:,j]
            theta_temp[j] = theta[j] - alpha / m * np.sum(j_deriv)
        
        theta = theta_temp
        
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #



        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history
