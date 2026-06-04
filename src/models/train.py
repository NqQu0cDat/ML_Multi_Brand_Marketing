"""Model training module"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def train_model(X, y, model_type="xgboost", test_size=0.2, random_state=42):
    """
    Train a machine learning model
    
    Args:
        X (pd.DataFrame or np.ndarray): Features
        y (pd.Series or np.ndarray): Target variable
        model_type (str): Type of model to train
        test_size (float): Test set size
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Contains trained model, test data, and performance metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Select model
    if model_type == "xgboost":
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            random_state=random_state
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state
        )
    else:  # linear_regression
        model = LinearRegression()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    }
    
    # Print results
    print(f"\n{model_type.upper()} Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return results

def save_model(model, filepath):
    """
    Save trained model to file
    
    Args:
        model: Trained model
        filepath (str): Path to save model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load trained model from file
    
    Args:
        filepath (str): Path to model file
        
    Returns:
        Trained model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
