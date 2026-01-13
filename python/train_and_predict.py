import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import pickle
import os
import json
from datetime import datetime

np.random.seed(42)

# Convert all salaries to yearly units (HOURLY * 2080, MONTHLY * 12, YEARLY unchanged)
# Creates SalaryMid_Normalized column without overwriting original SalaryMid
def normalize_salary_to_yearly(df):
    df_normalized = df.copy()
    
    # Check if CompensationType and salary columns exist
    if 'CompensationType' not in df_normalized.columns:
        print("Warning: CompensationType column not found. Skipping normalization.")
        return df_normalized
    
    if 'SalaryMid' not in df_normalized.columns:
        print("Warning: SalaryMid column not found. Skipping normalization.")
        return df_normalized
    
    # Handle NaN values - fill with 'YEARLY' as default
    df_normalized['CompensationType'] = df_normalized['CompensationType'].fillna('YEARLY')
    
    # Convert to uppercase for consistent matching
    comp_type_upper = df_normalized['CompensationType'].astype(str).str.upper()
    
    # Count before normalization for reporting
    hourly_count = (comp_type_upper == 'HOURLY').sum()
    monthly_count = (comp_type_upper == 'MONTHLY').sum()
    yearly_count = (comp_type_upper == 'YEARLY').sum()
    
    df_normalized['SalaryMid_Normalized'] = df_normalized['SalaryMid']
    
    mask_hourly = comp_type_upper == 'HOURLY'
    mask_monthly = comp_type_upper == 'MONTHLY'
    
    # 2080 = 40 hours/week * 52 weeks/year
    if mask_hourly.any():
        df_normalized.loc[mask_hourly, 'SalaryMid_Normalized'] = df_normalized.loc[mask_hourly, 'SalaryMid'] * 2080
    
    # Monthly to yearly conversion
    if mask_monthly.any():
        df_normalized.loc[mask_monthly, 'SalaryMid_Normalized'] = df_normalized.loc[mask_monthly, 'SalaryMid'] * 12
    
    # Report normalization
    print(f"\nSalary Normalization Summary:")
    print(f"  Hourly salaries normalized: {hourly_count} (×2080)")
    print(f"  Monthly salaries normalized: {monthly_count} (×12)")
    print(f"  Yearly salaries (unchanged): {yearly_count}")
    print(f"  All salaries now in YEARLY units for training (stored in SalaryMid_Normalized)")
    
    return df_normalized

# Load job postings and skills CSVs, merge them, create binary skill features, normalize salaries
# Returns: (merged dataframe, list of all unique skills)
def load_and_prepare_data(job_postings_path, skills_path):
    print("Loading REAL DATA from CSV files...")
    print(f"  Job postings: {job_postings_path}")
    print(f"  Skills: {skills_path}")
    df_jobs = pd.read_csv(job_postings_path)
    df_skills = pd.read_csv(skills_path)
    
    if len(df_jobs) == 0:
        raise ValueError("ERROR: No real data found in job postings file. Cannot proceed without real data.")
    if len(df_skills) == 0:
        raise ValueError("ERROR: No real data found in skills file. Cannot proceed without real data.")
    
    print(f"  Loaded {len(df_jobs)} REAL job postings from CSV")
    print(f"  Loaded {len(df_skills)} REAL skill records from CSV")
    
    # Group skills by PostingID and join into comma-separated string
    skills_grouped = df_skills.groupby('PostingID')['Skills'].apply(
        lambda x: ','.join(x.astype(str))
    ).reset_index()
    skills_grouped.columns = ['PostingID', 'AllSkills']
    
    # Merge skills into job postings dataframe
    df = df_jobs.merge(skills_grouped, on='PostingID', how='left')
    
    # Extract all unique skills from comma-separated strings
    all_skills = set()
    for skills_str in df['AllSkills'].dropna():
        all_skills.update([s.strip() for s in str(skills_str).split(',')])
    
    # Create binary features: Has_Python, Has_Java, etc. (1 if skill present, 0 otherwise)
    for skill in all_skills:
        if skill and len(skill) > 0:
            df[f'Has_{skill}'] = df['AllSkills'].apply(
                lambda x: 1 if pd.notna(x) and skill in str(x) else 0
            )
    
    # Normalize all salaries to yearly units before returning
    df = normalize_salary_to_yearly(df)
    
    return df, list(all_skills)

# Encode categorical columns to numeric using LabelEncoder
# Creates JobTitle_Encoded, RoleLevel_Encoded, etc. columns
# Returns: (encoded dataframe, dict of encoders for later inverse_transform)
def encode_categorical_features(df):
    df_encoded = df.copy()
    
    encoders = {}
    categorical_cols = ['JobTitle', 'RoleLevel', 'Location', 'Industry', 'RemoteType']
    
    # Encode each categorical column and save encoder for predictions
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[f'{col}_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders

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

# Generate salary and compensation type predictions for all rows in df
# Creates predictions CSV with PredictedSalary, ActualSalaryYearly, Industry, RoleLevel, etc.
# Returns: predictions DataFrame
def generate_predictions(df, salary_model, comp_model, feature_cols, encoders, comp_encoder):
    print(f"Generating predictions for {len(df)} REAL job postings...")
    
    # Prepare feature matrix: get existing columns, add missing ones as zeros
    X = df[[col for col in feature_cols if col in df.columns]].copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols].fillna(0)
    
    # Predict salaries using trained model
    print("  Predicting salaries...")
    pred_salaries = salary_model.predict(X)
    
    # Calculate confidence intervals using sample of trees (faster than all trees)
    # Use std dev across tree predictions to estimate uncertainty
    print("  Calculating confidence intervals...")
    num_trees = len(salary_model.estimators_)
    num_sample = min(20, num_trees)
    step = max(1, num_trees // num_sample)
    sample_trees = np.arange(0, num_trees, step)[:num_sample]
    tree_preds_all = np.array([salary_model.estimators_[i].predict(X) for i in sample_trees])
    std_devs = np.std(tree_preds_all, axis=0)
    
    # 95% confidence interval: mean ± 1.96 * std_dev
    pred_lowers = np.maximum(0, pred_salaries - 1.96 * std_devs)
    pred_uppers = pred_salaries + 1.96 * std_devs
    
    # Predict compensation type and get confidence scores
    print("  Predicting compensation types...")
    comp_probas = comp_model.predict_proba(X)
    comp_preds = comp_model.predict(X)
    comp_types = comp_encoder.inverse_transform(comp_preds)
    comp_confidences = np.max(comp_probas, axis=1)
    
    # Calculate overall confidence: combine salary confidence (based on std dev) and comp type confidence
    salary_confidences = np.maximum(0.5, np.minimum(0.95, 1 - (std_devs / np.maximum(pred_salaries, 1))))
    overall_confidences = (salary_confidences + comp_confidences) / 2
    
    print("  Building predictions dataframe (linked to REAL PostingIDs)...")
    
    if 'Industry' not in df.columns:
        raise ValueError("ERROR: Industry column missing from input data. Cannot generate predictions without Industry.")
    if 'RoleLevel' not in df.columns:
        raise ValueError("ERROR: RoleLevel column missing from input data. Cannot generate predictions without RoleLevel.")
    
    # Use original uncapped normalized salary if available (before outlier capping)
    # This shows the real actual salary, not the capped version used for training
    if 'SalaryMid_Normalized_Original' in df.columns:
        actual_salary_col = 'SalaryMid_Normalized_Original'
        print("  Using original uncapped normalized salaries for ActualSalaryYearly")
    elif 'SalaryMid_Normalized' in df.columns:
        actual_salary_col = 'SalaryMid_Normalized'
        print("  WARNING: Using capped normalized salaries (original not available)")
    else:
        raise ValueError("ERROR: SalaryMid_Normalized column missing. Cannot generate predictions without normalized actual salary.")
    
    # Build predictions DataFrame with all required columns for dashboard
    predictions = pd.DataFrame({
        'PostingID': df['PostingID'].values,
        'PredictedSalary': np.round(pred_salaries).astype(int),
        'PredictedSalaryLower': np.round(pred_lowers).astype(int),
        'PredictedSalaryUpper': np.round(pred_uppers).astype(int),
        'ActualSalaryYearly': np.round(df[actual_salary_col].values).astype(int),
        'Industry': df['Industry'].values,
        'RoleLevel': df['RoleLevel'].values,
        'PredictedCompType': comp_types,
        'PredictedCompTypeConfidence': np.round(comp_confidences, 3),
        'ConfidenceScore': np.round(overall_confidences, 3),
        'ModelVersion': 1.0
    })
    
    assert len(predictions) == len(df), "ERROR: Predictions must match input data length"
    assert all(predictions['PostingID'].isin(df['PostingID'])), "ERROR: All PostingIDs must be from real data"
    
    print(f"  Predictions include: Industry, RoleLevel, ActualSalaryYearly (normalized yearly)")
    
    return predictions

# Main training pipeline: load data, normalize salaries, cap outliers, train models, generate predictions
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # File paths
    job_postings_path = os.path.join(base_dir, 'data', 'transformed_job_postings.csv')
    skills_path = os.path.join(base_dir, 'data', 'transformed_skills.csv')
    predictions_path = os.path.join(base_dir, 'data', 'transformed_predictions.csv')
    model_path = os.path.join(base_dir, 'python', 'salary_model.pkl')
    importance_path = os.path.join(base_dir, 'python', 'feature_importance.json')
    
    print("=" * 60)
    print("FutureWorks Salary & Compensation Prediction Model")
    print("=" * 60)
    
    df, all_skills = load_and_prepare_data(job_postings_path, skills_path)
    print(f"Loaded {len(df)} REAL job postings with {len(all_skills)} unique skills from REAL data")
    
    if 'SalaryMid' not in df.columns:
        raise ValueError("ERROR: SalaryMid column missing. Cannot train without real salary data.")
    real_salary_count = df['SalaryMid'].notna().sum()
    if real_salary_count == 0:
        raise ValueError("ERROR: No real salary data found. Cannot train without real salary values.")
    print(f"  Validated: {real_salary_count} job postings have REAL salary data")
    
    # Validate normalized column exists after normalization
    if 'SalaryMid_Normalized' not in df.columns:
        raise ValueError("ERROR: SalaryMid_Normalized column missing after normalization. Cannot train without normalized salaries.")
    real_normalized_count = df['SalaryMid_Normalized'].notna().sum()
    if real_normalized_count == 0:
        raise ValueError("ERROR: No normalized salary data found. Cannot train without normalized salary values.")
    print(f"  Validated: {real_normalized_count} job postings have REAL normalized salary data")
    
    # Encode categorical features to numeric
    df_encoded, encoders = encode_categorical_features(df)
    print("Features encoded (deterministic encoding of REAL categorical data)")
    
    # Save original normalized salaries BEFORE capping (needed for ActualSalaryYearly in predictions CSV)
    if 'SalaryMid_Normalized' in df_encoded.columns:
        df_encoded['SalaryMid_Normalized_Original'] = df_encoded['SalaryMid_Normalized'].copy()
        print("\nSaved original normalized salaries (before capping) for predictions CSV")
    
    # Cap extreme salary outliers to prevent model from learning unrealistic values
    # Range: $20,000 - $400,000 yearly (reasonable salary bounds)
    if 'SalaryMid_Normalized' in df_encoded.columns:
        print("\nCapping salary outliers (20,000 - 400,000 range)...")
        before_cap = df_encoded['SalaryMid_Normalized'].describe()
        df_encoded['SalaryMid_Normalized'] = df_encoded['SalaryMid_Normalized'].clip(20000, 400000)
        after_cap = df_encoded['SalaryMid_Normalized'].describe()
        capped_count = ((df_encoded['SalaryMid_Normalized'] == 20000) | (df_encoded['SalaryMid_Normalized'] == 400000)).sum()
        print(f"  Capped {capped_count} extreme outliers")
        print(f"  Salary range after capping: ${df_encoded['SalaryMid_Normalized'].min():,.0f} - ${df_encoded['SalaryMid_Normalized'].max():,.0f}")
        print(f"  NOTE: Original uncapped salaries saved for ActualSalaryYearly in predictions CSV")
    
    if 'SalaryMid_Normalized' in df_encoded.columns:
        print("\nSalary Normalization Sanity Check (sample of 20 rows):")
        print(df_encoded[['SalaryMid', 'CompensationType', 'SalaryMid_Normalized', 'SalaryMid_Normalized_Original']].sample(20, random_state=42))
        print()
    
    # Train both models using CAPPED salaries (outliers removed)
    salary_model, salary_features, salary_metrics, salary_importance = train_salary_model(df_encoded)
    comp_model, comp_features, comp_encoder, comp_metrics, comp_importance = train_compensation_type_model(df_encoded)
    
    # Generate predictions (will use ORIGINAL uncapped salaries for ActualSalaryYearly)
    print("\nGenerating predictions...")
    predictions_df = generate_predictions(
        df_encoded, salary_model, comp_model,
        salary_features, encoders, comp_encoder
    )
    
    # Save predictions CSV for dashboard backend
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n[OK] Predictions saved to: {predictions_path}")
    print(f"  Generated {len(predictions_df)} predictions")
    
    # Save trained models with encoders and metadata
    model_data = {
        'salary_model': salary_model,
        'comp_model': comp_model,
        'salary_features': salary_features,
        'comp_features': comp_features,
        'encoders': encoders,
        'comp_encoder': comp_encoder,
        'metrics': {
            'salary': salary_metrics,
            'compensation_type': comp_metrics
        },
        'trained_date': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"[OK] Models saved to: {model_path}")
    
    # Save top 20 most important features for each model
    importance_data = {
        'salary_model': {
            'top_features': sorted(
                salary_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        },
        'compensation_type_model': {
            'top_features': sorted(
                comp_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        }
    }
    
    os.makedirs(os.path.dirname(importance_path), exist_ok=True)
    with open(importance_path, 'w') as f:
        json.dump(importance_data, f, indent=2)
    print(f"[OK] Feature importance saved to: {importance_path}")
    
    print("\n" + "=" * 60)
    print("Model Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()



