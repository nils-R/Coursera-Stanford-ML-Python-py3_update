import numpy as np


def sigmoid(z):
    """computes the sigmoid of z."""
    g = 1 / (1 + np.exp(-z) )
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).
# =============================================================
    return g