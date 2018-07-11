# -*- coding: utf-8 -*-
"""
Created on Mon Jul 9 2018

@author: N12667
"""

import unittest
import pandas as pd
import numpy as np
from scipy.io import loadmat
import random
import sys, os

class Ex3Test(unittest.TestCase):
    def setUp(self):
        pass
             
    def tearDown(self):
        pass
        
    def test_case1(self):
        self.X = np.column_stack((np.ones(5), np.arange(-5,5).reshape(2,5).T))
        self.y = np.arange(-2,3)
        self.Xval= np.row_stack((self.X, self.X)).reshape(10,3) / 10
        self.yval= np.ravel(np.row_stack((self.y, self.y)).reshape(10,1)) / 10
        correct_train = np.array([0.000000, 0.031250, 0.013333, 0.005165, 0.002268])
        correct_val = np.array([3.0000e-002, 5.3125e-003, 6.0000e-004, 9.2975e-005, 2.2676e-005])
        
        error_train, error_val = learningCurve(self.X, self.y, self.Xval, self.yval, 1)
        
        print(error_train, correct_train, thetas)
        self.assertEqual(error_train, correct_train)
        self.assertEqual(error_val, correct_val)
  
               
 
if __name__ == '__main__':  
    unittest.main()