"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distribution(df, column, bins=30, title=None):
    """
    Plot distribution of a column
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name
        bins (int): Number of bins
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(title or f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_correlation(df, figsize=(12, 8)):
    """
    Plot correlation heatmap
    
    Args:
        df (pd.DataFrame): Input dataframe (numeric columns only)
        figsize (tuple): Figure size
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    plt.figure(figsize=figsize)
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_categorical(df, column, figsize=(10, 6)):
    """
    Plot categorical distribution
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Categorical column name
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    value_counts = df[column].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
