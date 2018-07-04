# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 07:25:29 2018

@author: N12667
"""

import unittest
import pandas as pd
import numpy as np

from costFunction import costFunction

class Ex2Test(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt('ex2data1.txt', delimiter=',')           
        m = data.shape[0]
        # Add intercept term to x and X_test        
        data = np.concatenate((np.ones((m, 1)), data), axis=1)  
        n = data.shape[1]

        self.X = data[:,0:n-1] 
        self.y = data[:, n-1]
        self.initial_theta = np.zeros(n-1)
         
    def tearDown(self):
        pass
        
    def test_X_length(self):
        self.assertGreater(self.X.shape[0], 0)
    
    def test_X_width(self):
        self.assertEqual(self.X.shape[1], 3)
        
    def test_cost_function(self): 
        correct = 0.69314718055994529
        estimated = costFunction(self.initial_theta, self.X, self.y)
        self.assertAlmostEqual(correct, estimated)
 
if __name__ == '__main__':        
    unittest.main()