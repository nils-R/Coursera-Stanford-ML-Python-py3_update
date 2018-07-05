import matplotlib.pyplot as plt
import numpy as np


def plotData(x, y):
    """
    plots the data points and gives the figure axes labels of
    population and profit.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the
#               population and revenue data have been passed in
#               as the x and y arguments of this function.

    plt.figure()  # open a new figure window
    plt.plot(x, y, 'rx', markersize=6)#linestyle='None')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    #return plt
    
# ============================================================

# =============================================================================
# data = np.loadtxt('ex1data1.txt', delimiter=',')
# x=data[:, 0]
# y=data[:, 1]
# plty = plotData(x, y)
# plty.show()
# =============================================================================
