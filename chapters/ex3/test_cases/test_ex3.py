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
try:
    from definitions import ROOT_DIR
except(ImportError):
    if 'N12667' in os.getcwd():
        ROOT_DIR = 'C:\\Users\\N12667\\PythonScripts\\Training\\Coursera-Stanford-ML-Python'
    elif 'Jenkins' in os.getcwd():
        ROOT_DIR = 'C:\\Program Files (x86)\\Jenkins\\workspace\\stanford_machine_learning'

import sys, os
if os.path.abspath(os.path.join(ROOT_DIR,'chapters','ex2')) not in sys.path:
     sys.path.append(os.path.join(ROOT_DIR,'chapters','ex2'))
from lrCostFunction import lrCostFunction
from sigmoid import sigmoid

class Ex3Test(unittest.TestCase):
    def setUp(self):
        data = loadmat(os.path.join(ROOT_DIR,'chapters','ex3','ex3data1.mat'))
             
        X = data['X']
        m, n = X.shape
        #X = np.concatenate((np.ones((m, 1)), X), axis=1)
        
        y = data['y']

        self.X = X 
        self.y = y
        self.initial_theta =  (np.random.rand(n,1)-np.random.rand(n,1))/100
        #self.theta = # random array
        self.Lambda = 0.1 
         
    def tearDown(self):
        pass
        
    def test_X_theta_shapes(self):
        self.assertEqual(self.X.shape[1], self.initial_theta.shape[0])
    
    def test_X_width(self):
        self.assertGreater(self.X.shape[1], 0)
        
    def test_cost_lrfunction0(self): 
        correct = 0.69314718055994529
        estimated = lrCostFunction(self.initial_theta, self.X, self.y, self.Lambda)[0]
        self.assertAlmostEqual(correct, estimated)
               
 
if __name__ == '__main__':  
    import xmlrunner
    unittest.main()