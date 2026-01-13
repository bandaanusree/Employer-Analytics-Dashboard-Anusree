# FutureWorks Salary & Compensation Dashboard

A modern, interactive web dashboard for salary and compensation analytics with AI-powered predictions.

## 🚀 Features

- **AI-Powered Predictions**: Salary and compensation type predictions using machine learning
- **Interactive Visualizations**: Beautiful charts and graphs using Recharts
- **Real-time Analytics**: Live data analysis and insights
- **Benchmarking**: Compare employer offers against market standards
- **Responsive Design**: Works on desktop, tablet, and mobile

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## 🛠️ Installation

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Node Dependencies

```bash
cd frontend
npm install
```

### 3. Transform Real Data and Train Models

```bash
# Transform real data from "Real Data" folder
cd scripts
python transform_real_data.py

# Train ML models on real data
cd ../python
python train_and_predict.py
```

This will:
- Transform real data from `Real Data/` folder to `data/` folder
- Generate predictions using ML models trained on real data
- Save trained models to `python/salary_model.pkl`

## 🚀 Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
python app.py
```

Backend will run on `http://localhost:5000`

### Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

Frontend will run on `http://localhost:3000`

## 📁 Project Structure

```
dashboard/
├── backend/              # Flask API
│   ├── app.py           # Main API server
│   └── requirements.txt # Python dependencies
├── frontend/             # React frontend
│   ├── src/
│   │   ├── pages/       # Dashboard pages
│   │   ├── components/  # Reusable components
│   │   └── api/         # API client
│   └── package.json
├── data/                 # CSV data files
└── python/              # ML models
```

## 🎯 Dashboard Pages

1. **Home** - Overview with KPIs and key metrics
2. **Salary Overview** - Market salary insights
3. **Predictions** - AI predictions and accuracy analysis
4. **Benchmarking** - Employer vs Market comparison
5. **Compensation Types** - Hourly vs Yearly analysis

## 🔧 API Endpoints

- `GET /api/job-postings` - Get job postings (with filters)
- `GET /api/predictions` - Get predictions
- `GET /api/analytics/salary-summary` - Salary statistics
- `GET /api/analytics/prediction-accuracy` - Model accuracy metrics
- `GET /api/analytics/benchmarking` - Benchmarking data
- `POST /api/predict` - Predict salary for new posting

## 🎨 Technologies

- **Frontend**: React, Recharts, Axios
- **Backend**: Flask, Pandas, scikit-learn
- **Visualization**: Recharts (React charting library)

## 📊 Model Performance

- **Salary Prediction**: R² = 0.92, MAE = $7,051
- **Compensation Type**: 100% accuracy

## 🐛 Troubleshooting

**Backend not starting?**
- Check Python version (3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Ensure data files exist in `data/` folder

**Frontend not loading?**
- Check Node version (16+)
- Install dependencies: `npm install`
- Check API connection (backend running on port 5000)

**No data showing?**
- Ensure CSV files are in `data/` folder
- Check backend logs for errors
- Verify API endpoints are responding

## 📝 License

This project is provided as-is for FutureWorks Employability Analytics.

---

**Built with ❤️ for data-driven compensation decisions**



