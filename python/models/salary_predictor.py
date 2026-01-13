import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Train RandomForestRegressor to predict salary (in yearly units)
# Uses encoded categorical features + binary skill features
# Returns: (trained model, feature column names, metrics dict, feature importance dict)
def train_salary_model(df, target_col='SalaryMid_Normalized'):
    # Feature columns: encoded categoricals + all Has_* skill binary features
    feature_cols = [
        'JobTitle_Encoded', 'RoleLevel_Encoded', 'Location_Encoded',
        'Industry_Encoded', 'RemoteType_Encoded'
    ] + [col for col in df.columns if col.startswith('Has_')]
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("No valid data for training")
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # RandomForest parameters: 100 trees, max depth 15, min 5 samples to split
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Calculate predictions and metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_mape': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    
    print(f"\nSalary Model Performance:")
    print(f"  Training MAE: ${metrics['train_mae']:,.2f}")
    print(f"  Test MAE: ${metrics['test_mae']:,.2f}")
    print(f"  Test MAPE: {metrics['test_mape']:.2f}%")
    print(f"  Test R2: {metrics['test_r2']:.4f}")
    
    return model, feature_cols, metrics, feature_importance

