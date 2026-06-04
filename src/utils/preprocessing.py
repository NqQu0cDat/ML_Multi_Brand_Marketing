"""Data preprocessing utilities"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, target_column=None, drop_columns=None):
    """
    Basic preprocessing of dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Target column name for prediction
        drop_columns (list): Columns to drop
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Remove duplicate rows
    df_processed = df_processed.drop_duplicates()
    print(f"Removed {len(df) - len(df_processed)} duplicate rows")
    
    # Drop specified columns
    if drop_columns:
        df_processed = df_processed.drop(columns=[col for col in drop_columns if col in df_processed.columns])
    
    # Handle missing values
    print("\nMissing values:")
    print(df_processed.isnull().sum())
    
    # Fill numeric columns with median, categorical with mode
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    return df_processed

def encode_categorical(df, categorical_columns):
    """
    Encode categorical columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list): List of categorical column names
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical columns
    """
    df_encoded = df.copy()
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded

def scale_features(X, scaler=None):
    """
    Scale numeric features
    
    Args:
        X (pd.DataFrame): Feature dataframe
        scaler: StandardScaler instance (if None, creates new)
        
    Returns:
        np.ndarray: Scaled features
        StandardScaler: Fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler
