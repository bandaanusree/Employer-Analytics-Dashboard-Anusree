import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Train RandomForestClassifier to predict compensation type (Hourly/Yearly/Monthly)
# Returns: (trained model, feature column names, label encoder, metrics dict, feature importance dict)
def train_compensation_type_model(df):
    # Same features as salary model
    feature_cols = [
        'JobTitle_Encoded', 'RoleLevel_Encoded', 'Location_Encoded',
        'Industry_Encoded', 'RemoteType_Encoded'
    ] + [col for col in df.columns if col.startswith('Has_')]
    
    X = df[feature_cols].fillna(0)
    y = df['CompensationType']
    
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Encode compensation type labels (Hourly -> 0, Yearly -> 1, etc.)
    le_comp = LabelEncoder()
    y_encoded = le_comp.fit_transform(y)
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    metrics = {
        'train_accuracy': train_acc * 100,
        'test_accuracy': test_acc * 100
    }
    
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    
    print(f"\nCompensation Type Model Performance:")
    print(f"  Training Accuracy: {metrics['train_accuracy']:.2f}%")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.2f}%")
    
    return model, feature_cols, le_comp, metrics, feature_importance

