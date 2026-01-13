import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime

# Import from organized modules
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

from models.salary_predictor import train_salary_model
from models.compensation_predictor import train_compensation_type_model
from preprocessing.feature_engineering import load_and_prepare_data, encode_categorical_features
from training.predictor import generate_predictions

np.random.seed(42)

# Main training pipeline: load data, normalize salaries, cap outliers, train models, generate predictions
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # File paths
    job_postings_path = os.path.join(base_dir, 'data', 'transformed_job_postings.csv')
    skills_path = os.path.join(base_dir, 'data', 'transformed_skills.csv')
    predictions_path = os.path.join(base_dir, 'data', 'transformed_predictions.csv')
    model_path = os.path.join(base_dir, 'python', 'saved_models', 'salary_model.pkl')
    importance_path = os.path.join(base_dir, 'python', 'saved_models', 'feature_importance.json')
    
    print("=" * 60)
    print("FutureWorks Salary & Compensation Prediction Model")
    print("=" * 60)
    
    # Load and prepare data
    df, all_skills = load_and_prepare_data(job_postings_path, skills_path)
    print(f"Loaded {len(df)} job postings with {len(all_skills)} unique skills")
    
    # Encode categorical features to numeric
    df_encoded, encoders = encode_categorical_features(df)
    print("Features encoded")
    
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

