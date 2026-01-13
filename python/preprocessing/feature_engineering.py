import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

