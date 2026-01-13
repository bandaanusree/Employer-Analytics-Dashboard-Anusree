import pandas as pd
import numpy as np

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

