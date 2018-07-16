import numpy as np


def recoverData(Z, U, K):
    """
    recovers an approximation the
    original data that has been reduced to K dimensions. It returns the
    approximate reconstruction in X_rec.
    """
    if K == 1:
        recovered_j = U[:,K][:,None].dot(Z.T)
    else:
        recovered_j = U[:,K].dot(Z.T)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the approximation of the data by projecting back
    #               onto the original space using the top K eigenvectors in U.
    #
    #               For the i-th example Z(i,:), the (approximate)
    #               recovered data for dimension j is given as follows:
    #                    v = Z(i, :)'
    #                    recovered_j = v' * U(j, 1:K)'
    #
    #               Notice that U(j, 1:K) is a row vector.
    #
    # =============================================================

    return recovered_j#X_rec
