"""Prediction module"""

import numpy as np
import pandas as pd

def make_prediction(model, X):
    """
    Make predictions using trained model
    
    Args:
        model: Trained model
        X (pd.DataFrame or np.ndarray): Feature data
        
    Returns:
        np.ndarray: Predictions
    """
    predictions = model.predict(X)
    return predictions

def predict_batch(model, data_list):
    """
    Make predictions on multiple samples
    
    Args:
        model: Trained model
        data_list (list): List of feature arrays
        
    Returns:
        list: List of predictions
    """
    predictions = []
    for data in data_list:
        pred = model.predict(data.reshape(1, -1))[0]
        predictions.append(pred)
    
    return predictions
