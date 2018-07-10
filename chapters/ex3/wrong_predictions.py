# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 09:42:38 2018

@author: N12667
"""
import pandas as pd

def wrong_preds(y, p, predictions):
    y = pd.DataFrame(y, columns=['correct_answer'])
    p = pd.DataFrame(p, columns=['predicted_answer'])
    probnames = [string + str(val) for string, val in zip( ('probability: ',)*predictions.shape[1], range(1,predictions.shape[1]+1))]
    predictions = pd.DataFrame(predictions, columns=probnames)        
    
    y_diff = y[p['predicted_answer']!=y['correct_answer']]
    mask = y_diff.index
    p_diff= p.iloc[mask,:]
    predictions_diff = predictions.iloc[mask,:]
    
    wrong_predictions = y_diff.join([p_diff, predictions_diff])
    
    correct_value_probability_column = [string + str(val) for string, val in zip( ('probability: ',)*wrong_predictions.shape[0], wrong_predictions['correct_answer'])]
    correct_value_probability = [wrong_predictions[label].iloc[index] for index, label in enumerate(correct_value_probability_column)]
    error_margin = predictions_diff.max(axis=1) - correct_value_probability
    
    wrong_predictions.insert(loc=2, column='error_margin', value=error_margin)
    wrong_predictions['predicted_answer'] = wrong_predictions['predicted_answer'].astype(str)
    return wrong_predictions