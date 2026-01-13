from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)

# Load data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Global data storage
job_postings = None
skills = None
predictions = None
employer_offers = None
model_data = None

def load_data():
    global job_postings, skills, predictions, employer_offers, model_data
    
    try:
        job_postings = pd.read_csv(os.path.join(DATA_DIR, 'transformed_job_postings.csv'))
        skills = pd.read_csv(os.path.join(DATA_DIR, 'transformed_skills.csv'))
        predictions = pd.read_csv(os.path.join(DATA_DIR, 'transformed_predictions.csv'))
        employer_offers = pd.read_csv(os.path.join(DATA_DIR, 'transformed_employer_offers.csv'))
        
        # Load model if available
        model_path = os.path.join(BASE_DIR, 'python', 'salary_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        
        print("Data loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

# Load data on startup
load_data()

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'data_loaded': job_postings is not None,
        'model_loaded': model_data is not None
    })

# Apply common filters from query parameters (industry, experience_level, compensation_type)
def apply_filters(df):
    industry = request.args.get('industry')
    experience_level = request.args.get('experience_level')
    compensation_type = request.args.get('compensation_type')
    
    if industry:
        df = df[df['Industry'] == industry]
    if experience_level:
        df = df[df['RoleLevel'] == experience_level]
    if compensation_type:
        df = df[df['CompensationType'] == compensation_type]
    
    return df

@app.route('/api/job-postings', methods=['GET'])
def get_job_postings():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    if date_from:
        df = df[pd.to_datetime(df['PostedDate']) >= pd.to_datetime(date_from)]
    if date_to:
        df = df[pd.to_datetime(df['PostedDate']) <= pd.to_datetime(date_to)]
    
    return jsonify(df.to_dict('records'))

# Get individual predictions for scatter plot (PredictedSalary vs ActualSalaryYearly)
# Returns filtered predictions with Industry, RoleLevel for frontend visualization
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    if predictions is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        df = predictions.copy()
        
        if 'PredictedSalary' not in df.columns or 'ActualSalaryYearly' not in df.columns:
            return jsonify({'error': 'Missing required columns: PredictedSalary or ActualSalaryYearly'}), 500
        
        # Merge with job_postings if Industry/RoleLevel missing from predictions CSV
        if 'Industry' not in df.columns and job_postings is not None and 'PostingID' in job_postings.columns:
            merge_cols = ['PostingID']
            if 'Industry' in job_postings.columns:
                merge_cols.append('Industry')
            if 'RoleLevel' in job_postings.columns:
                merge_cols.append('RoleLevel')
            if 'CompensationType' in job_postings.columns:
                merge_cols.append('CompensationType')
            
            df = df.merge(
                job_postings[merge_cols], 
                on='PostingID', 
                how='left'
            )
        
        if 'Industry' in df.columns or 'RoleLevel' in df.columns or 'CompensationType' in df.columns:
            df = apply_filters(df)
        
        # Convert to numeric and filter out invalid salaries (too low or too high)
        df['PredictedSalary'] = pd.to_numeric(df['PredictedSalary'], errors='coerce')
        df['ActualSalaryYearly'] = pd.to_numeric(df['ActualSalaryYearly'], errors='coerce')
        
        # Filter reasonable salary range: $10k - $500k yearly
        df = df[
            df['PredictedSalary'].notna() &
            df['ActualSalaryYearly'].notna() &
            (df['PredictedSalary'] >= 10000) & 
            (df['ActualSalaryYearly'] >= 10000) &
            (df['PredictedSalary'] <= 500000) &
            (df['ActualSalaryYearly'] <= 500000)
        ]
        
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({'error': f'Error processing predictions: {str(e)}'}), 500

@app.route('/api/skills', methods=['GET'])
def get_skills():
    if skills is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    posting_id = request.args.get('posting_id')
    if posting_id:
        filtered = skills[skills['PostingID'] == int(posting_id)]
        return jsonify(filtered.to_dict('records'))
    
    return jsonify(skills.to_dict('records'))

@app.route('/api/employer-offers', methods=['GET'])
def get_employer_offers():
    if employer_offers is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    return jsonify(employer_offers.to_dict('records'))

@app.route('/api/analytics/overview-kpis', methods=['GET'])
def get_overview_kpis():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    total_jobs = int(df['PostingID'].nunique()) if 'PostingID' in df.columns else int(len(df))
    
    highest_paying_industry = None
    highest_paying_salary = 0
    if 'Industry' in df.columns and 'SalaryMid' in df.columns:
        industry_df = df[df['Industry'].notna() & (df['Industry'] != 'Unknown') & (df['Industry'] != 'nan')]
        if len(industry_df) > 0:
            industry_salaries = industry_df.groupby('Industry')['SalaryMid'].mean().sort_values(ascending=False)
            if len(industry_salaries) > 0:
                highest_paying_industry = industry_salaries.index[0]
                highest_paying_salary = float(industry_salaries.iloc[0])
    
    yearly_df = df[df['CompensationType'] == 'Yearly'].copy() if 'CompensationType' in df.columns else df.copy()
    if 'SalaryMid' in yearly_df.columns:
        yearly_df = yearly_df[(yearly_df['SalaryMid'] >= 20000) & (yearly_df['SalaryMid'] <= 500000)]
        average_salary = float(yearly_df['SalaryMid'].mean()) if len(yearly_df) > 0 else 0
    else:
        average_salary = 0
    
    exp_mapping = {'Entry': 1, 'Junior': 1, 'Mid': 3, 'Senior': 4, 'Executive': 5, 'Lead': 4, 'Principal': 5}
    if 'RoleLevel' in df.columns:
        df['exp_numeric'] = df['RoleLevel'].map(exp_mapping).fillna(3)
        average_experience_level = float(df['exp_numeric'].mean()) if len(df) > 0 else 0
    else:
        average_experience_level = 0
    
    return jsonify({
        'total_jobs': total_jobs,
        'highest_paying_industry': highest_paying_industry or 'N/A',
        'highest_paying_salary': highest_paying_salary,
        'average_salary': average_salary,
        'average_experience_level': average_experience_level
    })

@app.route('/api/analytics/salary-summary', methods=['GET'])
def get_salary_summary():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    return jsonify({
        'median': float(df['SalaryMid'].median()),
        'average': float(df['SalaryMid'].mean()),
        'min': float(df['SalaryMin'].min()),
        'max': float(df['SalaryMax'].max()),
        'percentile_25': float(df['SalaryMid'].quantile(0.25)),
        'percentile_75': float(df['SalaryMid'].quantile(0.75)),
        'count': int(len(df))
    })

@app.route('/api/analytics/salary-insights-kpis', methods=['GET'])
def get_salary_insights_kpis():
    if job_postings is None or skills is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    yearly_df = df[df['CompensationType'] == 'Yearly'].copy() if 'CompensationType' in df.columns else df.copy()
    yearly_df = yearly_df[(yearly_df['SalaryMid'] >= 20000) & (yearly_df['SalaryMid'] <= 500000)]
    
    median_salary = float(yearly_df['SalaryMid'].median()) if len(yearly_df) > 0 else 0
    
    average_salary = float(yearly_df['SalaryMid'].mean()) if len(yearly_df) > 0 else 0
    
    highest_paying_industry = None
    highest_paying_industry_salary = 0
    if 'Industry' in df.columns and 'SalaryMid' in df.columns:
        industry_df = df[df['Industry'].notna() & (df['Industry'] != 'Unknown') & (df['Industry'] != 'nan')]
        if len(industry_df) > 0:
            industry_salaries = industry_df.groupby('Industry')['SalaryMid'].mean().sort_values(ascending=False)
            if len(industry_salaries) > 0:
                highest_paying_industry = industry_salaries.index[0]
                highest_paying_industry_salary = float(industry_salaries.iloc[0])
    
    highest_paying_skill = None
    highest_paying_skill_salary = 0
    
    filtered_posting_ids = df['PostingID'].unique() if 'PostingID' in df.columns else []
    
    if len(filtered_posting_ids) > 0:
        filtered_skills = skills[skills['PostingID'].isin(filtered_posting_ids)]
    else:
        filtered_skills = skills
    
    skill_salaries = {}
    for skill in filtered_skills['Skills'].unique():
        skill_postings = filtered_skills[filtered_skills['Skills'] == skill]['PostingID'].unique()
        avg_salary = df[df['PostingID'].isin(skill_postings)]['SalaryMid'].mean()
        if not pd.isna(avg_salary):
            skill_salaries[skill] = float(avg_salary)
    
    if skill_salaries:
        highest_paying_skill = max(skill_salaries, key=skill_salaries.get)
        highest_paying_skill_salary = skill_salaries[highest_paying_skill]
    
    return jsonify({
        'median_salary': median_salary,
        'average_salary': average_salary,
        'highest_paying_industry': highest_paying_industry or 'N/A',
        'highest_paying_industry_salary': highest_paying_industry_salary,
        'highest_paying_skill': highest_paying_skill or 'N/A',
        'highest_paying_skill_salary': highest_paying_skill_salary
    })

@app.route('/api/analytics/prediction-accuracy', methods=['GET'])
def get_prediction_accuracy():
    if predictions is None or job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    if 'ActualSalaryYearly' not in predictions.columns:
        return jsonify({'error': 'ActualSalaryYearly column missing from predictions'}), 500
    
    df = job_postings.copy()
    df = apply_filters(df)
    
    filtered_posting_ids = df['PostingID'].unique() if 'PostingID' in df.columns else []
    
    if len(filtered_posting_ids) > 0:
        filtered_predictions = predictions[predictions['PostingID'].isin(filtered_posting_ids)].copy()
    else:
        filtered_predictions = predictions.copy()
    
    if len(filtered_predictions) == 0:
        return jsonify({
            'mae': 0,
            'mape': 0,
            'r2': 0,
            'count': 0
        })
    
    filtered_predictions = filtered_predictions[
        (filtered_predictions['ActualSalaryYearly'] >= 10000) &
        (filtered_predictions['PredictedSalary'] >= 10000)
    ]
    
    if len(filtered_predictions) == 0:
        return jsonify({
            'mae': 0,
            'mape': 0,
            'r2': 0,
            'count': 0
        })
    
    merged = filtered_predictions[['PostingID', 'PredictedSalary', 'ActualSalaryYearly']].copy()
    merged['Actual'] = merged['ActualSalaryYearly']
    
    merged['Error'] = abs(merged['PredictedSalary'] - merged['Actual'])
    merged['ErrorPct'] = (merged['Error'] / merged['Actual']) * 100
    
    mae = float(merged['Error'].mean())
    mape = float(merged['ErrorPct'].mean())
    
    ss_res = ((merged['Actual'] - merged['PredictedSalary']) ** 2).sum()
    ss_tot = ((merged['Actual'] - merged['Actual'].mean()) ** 2).sum()
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
    
    return jsonify({
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'count': int(len(merged))
    })

@app.route('/api/analytics/prediction-kpis', methods=['GET'])
def get_prediction_kpis():
    if job_postings is None or predictions is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    yearly_df = df[df['CompensationType'] == 'Yearly'].copy() if 'CompensationType' in df.columns else df.copy()
    yearly_df = yearly_df[(yearly_df['SalaryMid'] >= 20000) & (yearly_df['SalaryMid'] <= 500000)]
    
    median_salary = float(yearly_df['SalaryMid'].median()) if len(yearly_df) > 0 else 0
    
    average_salary = float(yearly_df['SalaryMid'].mean()) if len(yearly_df) > 0 else 0
    
    filtered_posting_ids = df['PostingID'].unique() if 'PostingID' in df.columns else []
    if len(filtered_posting_ids) > 0:
        filtered_predictions = predictions[predictions['PostingID'].isin(filtered_posting_ids)].copy()
    else:
        filtered_predictions = predictions.copy()
    
    filtered_predictions = filtered_predictions[
        (filtered_predictions['PredictedSalary'] >= 20000) & 
        (filtered_predictions['PredictedSalary'] <= 500000) &
        (filtered_predictions['ActualSalaryYearly'] >= 20000) &
        (filtered_predictions['ActualSalaryYearly'] <= 500000)
    ]
    predicted_salary = float(filtered_predictions['PredictedSalary'].mean()) if len(filtered_predictions) > 0 else 0
    
    highest_paying_industry = None
    highest_paying_industry_salary = 0
    if 'Industry' in df.columns and 'SalaryMid' in df.columns:
        industry_df = df[df['Industry'].notna() & (df['Industry'] != 'Unknown') & (df['Industry'] != 'nan')]
        if len(industry_df) > 0:
            industry_salaries = industry_df.groupby('Industry')['SalaryMid'].mean().sort_values(ascending=False)
            if len(industry_salaries) > 0:
                highest_paying_industry = industry_salaries.index[0]
                highest_paying_industry_salary = float(industry_salaries.iloc[0])
    
    return jsonify({
        'median_salary': median_salary,
        'average_salary': average_salary,
        'predicted_salary': predicted_salary,
        'highest_paying_industry': highest_paying_industry or 'N/A',
        'highest_paying_industry_salary': highest_paying_industry_salary
    })

@app.route('/api/analytics/compensation-distribution', methods=['GET'])
def get_compensation_distribution():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    distribution = df['CompensationType'].value_counts().to_dict()
    total = len(df)
    
    result = {
        'distribution': {k: int(v) for k, v in distribution.items()},
        'percentages': {k: float((v / total) * 100) for k, v in distribution.items()},
        'total': int(total)
    }
    
    return jsonify(result)

@app.route('/api/analytics/salary-by-role', methods=['GET'])
def get_salary_by_role():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    result = df.groupby('JobTitle')['SalaryMid'].agg([
        ('median', 'median'),
        ('average', 'mean'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    result = result.sort_values('average', ascending=False)
    
    return jsonify(result.to_dict('records'))

@app.route('/api/analytics/salary-by-location', methods=['GET'])
def get_salary_by_location():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    result = df.groupby('Location')['SalaryMid'].agg([
        ('median', 'median'),
        ('average', 'mean'),
        ('count', 'count')
    ]).reset_index()
    
    result = result.sort_values('average', ascending=False)
    
    return jsonify(result.to_dict('records'))

@app.route('/api/analytics/salary-by-experience-level', methods=['GET'])
def get_salary_by_experience_level():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    df = df[df['RoleLevel'].notna() & (df['RoleLevel'] != 'nan')]
    
    result = df.groupby('RoleLevel')['SalaryMid'].agg([
        ('median', 'median'),
        ('average', 'mean'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    level_order = {'Junior': 1, 'Mid': 2, 'Senior': 3}
    result['sort_order'] = result['RoleLevel'].map(level_order).fillna(99)
    result = result.sort_values('sort_order').drop('sort_order', axis=1)
    
    return jsonify(result.to_dict('records'))

@app.route('/api/analytics/salary-by-industry', methods=['GET'])
def get_salary_by_industry():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    df = df[df['Industry'].notna() & (df['Industry'] != 'Unknown') & (df['Industry'] != 'nan')]
    
    result = df.groupby('Industry')['SalaryMid'].agg([
        ('median', 'median'),
        ('average', 'mean'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    result = result.sort_values('average', ascending=False)
    
    return jsonify(result.to_dict('records'))

@app.route('/api/analytics/salary-distribution', methods=['GET'])
def get_salary_distribution():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    df = df[(df['SalaryMid'] >= 20000) & (df['SalaryMid'] <= 500000)]
    
    bins = [0, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 250000, 300000, 500000]
    labels = ['$0-50K', '$50-75K', '$75-100K', '$100-125K', '$125-150K', 
              '$150-175K', '$175-200K', '$200-250K', '$250-300K', '$300K+']
    
    df['SalaryRange'] = pd.cut(df['SalaryMid'], bins=bins, labels=labels, include_lowest=True)
    
    distribution = df['SalaryRange'].value_counts().sort_index().to_dict()
    
    result = [{'range': str(k), 'count': int(v)} for k, v in distribution.items()]
    
    return jsonify(result)

@app.route('/api/analytics/salary-trends', methods=['GET'])
def get_salary_trends():
    if job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    df['PostedDate'] = pd.to_datetime(df['PostedDate'])
    df['YearMonth'] = df['PostedDate'].dt.to_period('M').astype(str)
    
    trends = df.groupby('YearMonth')['SalaryMid'].agg([
        ('median', 'median'),
        ('average', 'mean'),
        ('count', 'count')
    ]).reset_index()
    
    return jsonify(trends.to_dict('records'))

# Get prediction gaps aggregated by Industry (for "Top Prediction Gaps by Industry" table)
# Groups by Industry, calculates avg PredictedSalary and avg ActualSalaryYearly, computes gap metrics
# Categories: Overpaying (>+5%), Underpaying (<-5%), Competitive (between -5% and +5%)
@app.route('/api/analytics/prediction-gaps', methods=['GET'])
def get_prediction_gaps():
    if predictions is None or job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    required_cols = ['PostingID', 'PredictedSalary', 'ActualSalaryYearly', 'Industry', 'RoleLevel']
    missing_cols = [col for col in required_cols if col not in predictions.columns]
    if missing_cols:
        return jsonify({'error': f'Missing required columns in predictions: {missing_cols}'}), 500
    
    # Apply filters from query params to job_postings, then filter predictions to matching PostingIDs
    df = job_postings.copy()
    df = apply_filters(df)
    
    filtered_posting_ids = df['PostingID'].unique() if 'PostingID' in df.columns else []
    
    if len(filtered_posting_ids) > 0:
        filtered_predictions = predictions[predictions['PostingID'].isin(filtered_posting_ids)].copy()
    else:
        filtered_predictions = predictions.copy()
    
    if len(filtered_predictions) == 0:
        return jsonify([])
    
    # Filter out rows with invalid Industry values
    filtered_predictions = filtered_predictions[
        filtered_predictions['Industry'].notna() & 
        (filtered_predictions['Industry'] != '') & 
        (filtered_predictions['Industry'] != 'Unknown')
    ]
    
    if len(filtered_predictions) == 0:
        return jsonify([])
    
    # Filter out invalid salaries (< $10k yearly)
    filtered_predictions = filtered_predictions[
        (filtered_predictions['ActualSalaryYearly'] >= 10000) &
        (filtered_predictions['PredictedSalary'] >= 10000)
    ]
    
    if len(filtered_predictions) == 0:
        return jsonify([])
    
    # Helper: get most common role level per industry
    def get_most_common_role(x):
        mode_values = x.mode()
        if len(mode_values) > 0:
            return mode_values.iloc[0]
        elif len(x) > 0:
            return x.iloc[0]
        else:
            return 'Unknown'
    
    # Aggregate by Industry: mean PredictedSalary, mean ActualSalaryYearly, most common RoleLevel
    industry_gaps = filtered_predictions.groupby('Industry').agg({
        'PredictedSalary': 'mean',
        'ActualSalaryYearly': 'mean',
        'RoleLevel': get_most_common_role
    }).reset_index()
    
    industry_gaps.columns = ['Industry', 'PredictedSalary', 'ActualSalaryYearly', 'RoleLevel']
    
    # Calculate gap metrics: gap = actual - predicted, gapPct = (gap / actual) * 100
    industry_gaps['Gap'] = industry_gaps['ActualSalaryYearly'] - industry_gaps['PredictedSalary']
    industry_gaps['GapPct'] = (industry_gaps['Gap'] / industry_gaps['ActualSalaryYearly']) * 100
    
    # Categorize: >+5% = Overpaying, <-5% = Underpaying, else Competitive
    industry_gaps['Category'] = industry_gaps['GapPct'].apply(
        lambda x: 'Overpaying' if x > 5 else ('Underpaying' if x < -5 else 'Competitive')
    )
    
    # Sort by absolute gap % descending (biggest mismatches first)
    industry_gaps['AbsGapPct'] = industry_gaps['GapPct'].abs()
    industry_gaps = industry_gaps.sort_values(by='AbsGapPct', ascending=False)
    industry_gaps = industry_gaps.drop('AbsGapPct', axis=1)
    
    # Rename for frontend compatibility (frontend expects 'SalaryMid' but it's actually normalized yearly)
    industry_gaps = industry_gaps.rename(columns={'ActualSalaryYearly': 'SalaryMid'})
    
    return jsonify(industry_gaps.to_dict('records'))

@app.route('/api/analytics/benchmarking', methods=['GET'])
def get_benchmarking():
    if employer_offers is None or job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    market_data = job_postings.groupby(['JobTitle', 'Location'])['SalaryMid'].agg([
        ('market_median', 'median'),
        ('market_avg', 'mean'),
        ('market_p25', lambda x: x.quantile(0.25)),
        ('market_p75', lambda x: x.quantile(0.75)),
        ('count', 'count')
    ]).reset_index()
    
    benchmarking = employer_offers.merge(
        market_data,
        left_on=['Role', 'Location'],
        right_on=['JobTitle', 'Location'],
        how='left'
    )
    
    benchmarking['Gap'] = benchmarking['SalaryOffer'] - benchmarking['market_median']
    benchmarking['GapPct'] = (benchmarking['Gap'] / benchmarking['market_median']) * 100
    
    def calc_percentile(row):
        if pd.isna(row['market_median']):
            return None
        market_salaries = job_postings[
            (job_postings['JobTitle'] == row['Role']) &
            (job_postings['Location'] == row['Location'])
        ]['SalaryMid']
        if len(market_salaries) == 0:
            return None
        percentile = (market_salaries <= row['SalaryOffer']).sum() / len(market_salaries) * 100
        return float(percentile)
    
    benchmarking['MarketPercentile'] = benchmarking.apply(calc_percentile, axis=1)
    
    return jsonify(benchmarking.to_dict('records'))

@app.route('/api/analytics/top-skills', methods=['GET'])
def get_top_skills():
    if skills is None or job_postings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = job_postings.copy()
    
    # Apply filters
    df = apply_filters(df)
    
    filtered_posting_ids = df['PostingID'].unique() if 'PostingID' in df.columns else []
    
    if len(filtered_posting_ids) > 0:
        filtered_skills = skills[skills['PostingID'].isin(filtered_posting_ids)]
    else:
        filtered_skills = skills
    
    skill_counts = filtered_skills['Skills'].value_counts().head(20) if 'Skills' in filtered_skills.columns else pd.Series()
    
    skill_salaries = []
    for skill in skill_counts.index:
        skill_postings = filtered_skills[filtered_skills['Skills'] == skill]['PostingID'].unique()
        avg_salary = df[df['PostingID'].isin(skill_postings)]['SalaryMid'].mean()
        skill_salaries.append({
            'skill': skill,
            'frequency': int(skill_counts[skill]),
            'average_salary': float(avg_salary) if not pd.isna(avg_salary) else 0
        })
    
    return jsonify(skill_salaries)

@app.route('/api/predict', methods=['POST'])
def predict_salary():
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    
    try:
        job_title = data.get('job_title')
        location = data.get('location')
        role_level = data.get('role_level')
        
        similar = job_postings[
            (job_postings['JobTitle'] == job_title) &
            (job_postings['Location'] == location) &
            (job_postings['RoleLevel'] == role_level)
        ]
        
        if len(similar) > 0:
            pred_salary = float(similar['SalaryMid'].median())
            pred_lower = float(pred_salary * 0.85)
            pred_upper = float(pred_salary * 1.15)
        else:
            pred_salary = float(job_postings['SalaryMid'].median())
            pred_lower = float(pred_salary * 0.85)
            pred_upper = float(pred_salary * 1.15)
        
        return jsonify({
            'predicted_salary': pred_salary,
            'predicted_lower': pred_lower,
            'predicted_upper': pred_upper,
            'predicted_comp_type': 'Yearly',
            'confidence': 0.85
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/filters/industries', methods=['GET'])
def get_industries():
    if job_postings is None:
        return jsonify([])
    industries = job_postings['Industry'].unique().tolist()
    industries = [ind for ind in industries if ind and str(ind) != 'nan' and str(ind) != 'Unknown']
    return jsonify(sorted(industries))

@app.route('/api/filters/experience-levels', methods=['GET'])
def get_experience_levels():
    if job_postings is None:
        return jsonify([])
    levels = job_postings['RoleLevel'].unique().tolist()
    levels = [level for level in levels if level and str(level) != 'nan']
    return jsonify(sorted(levels))

@app.route('/api/filters/compensation-types', methods=['GET'])
def get_compensation_types():
    if job_postings is None:
        return jsonify(['Yearly', 'Hourly'])
    
    types = job_postings['CompensationType'].unique().tolist()
    types = [t for t in types if t and str(t) != 'nan']
    
    if 'Yearly' not in types:
        types.append('Yearly')
    if 'Hourly' not in types:
        types.append('Hourly')
    
    return jsonify(sorted(types))

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("Starting FutureWorks Dashboard API...")
    print(f"Data directory: {DATA_DIR}")
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, port=port, host='0.0.0.0')

