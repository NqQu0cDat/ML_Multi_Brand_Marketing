"""Data loading utilities"""

import pandas as pd
import os
from pathlib import Path

def load_data(file_path):
    """
    Load data from CSV file
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")
    return df

def load_all_datasets(data_dir="data/raw"):
    """
    Load all CSV datasets from a directory
    
    Args:
        data_dir (str): Directory path containing CSV files
        
    Returns:
        dict: Dictionary with brand names as keys and dataframes as values
    """
    datasets = {}
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            brand_name = file.replace('_campaign_data.csv', '')
            file_path = os.path.join(data_dir, file)
            datasets[brand_name] = load_data(file_path)
    
    return datasets
