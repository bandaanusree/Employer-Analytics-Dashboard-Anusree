import React, { useState, useEffect } from 'react'
import FilterBar from '../components/FilterBar'
import KPICard from '../components/KPICard'
import { getPredictionAccuracy, getPredictionGaps, getPredictions, getPredictionKPIs } from '../api/api'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell, ComposedChart, Line } from 'recharts'
import './Predictions.css'

function Predictions() {
  const [filters, setFilters] = useState({})
  const [kpis, setKpis] = useState(null)
  const [accuracy, setAccuracy] = useState(null)
  const [gaps, setGaps] = useState([])
  const [predictions, setPredictions] = useState([])  // Individual predictions for scatter plot
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadData()
  }, [filters])

  const loadData = async () => {
    setLoading(true)
    try {
      const [kpisRes, accuracyRes, gapsRes, predictionsRes] = await Promise.all([
        getPredictionKPIs(filters).catch(err => {
          console.error('Error loading KPIs:', err)
          return { data: null }
        }),
        getPredictionAccuracy(filters).catch(err => {
          console.error('Error loading accuracy:', err)
          return { data: null }
        }),
        getPredictionGaps(filters).catch(err => {
          console.error('Error loading gaps:', err)
          return { data: [] }
        }),
        getPredictions(filters).catch(err => {
          console.error('Error loading predictions:', err)
          return { data: [] }
        })
      ])
      
      setKpis(kpisRes.data)
      setAccuracy(accuracyRes.data)
      setGaps(gapsRes.data || [])
      setPredictions(predictionsRes.data || [])
    } catch (error) {
      console.error('Error loading data:', error)
      setError(error.message || 'Failed to load data')
      // Set defaults to prevent blank page
      setKpis(null)
      setAccuracy(null)
      setGaps([])
      setPredictions([])
    } finally {
      setLoading(false)
    }
  }

  // Show error if there's a critical error
  if (error && !kpis && !accuracy && gaps.length === 0 && predictions.length === 0) {
    return (
      <div className="predictions-page">
        <div className="page-header">
          <h1>Salary Predictions</h1>
        </div>
        <div style={{ padding: '20px', textAlign: 'center' }}>
          <p style={{ color: '#d62728' }}>Error: {error}</p>
          <p style={{ fontSize: '0.9em', color: '#666' }}>Please check the browser console for details.</p>
        </div>
      </div>
    )
  }

  // Categorize gaps - uses ALL filtered data from backend
  // Gaps are already aggregated by Industry and filtered by user's filter selections
  const gapCategories = {
    Overpaying: gaps.filter(g => g.Category === 'Overpaying').length,
    Competitive: gaps.filter(g => g.Category === 'Competitive').length,
    Underpaying: gaps.filter(g => g.Category === 'Underpaying').length
  }

  const categoryData = Object.entries(gapCategories).map(([name, value]) => ({ name, value }))

  // Scatter plot data - shows INDIVIDUAL predictions (Predicted vs Actual Salary)
  // Each point represents one job posting's predicted vs actual salary
  // Uses PredictedSalary and ActualSalaryYearly from transformed_predictions.csv
  // Filter out invalid salaries for display
  // Ensure predictions is always an array
  const predictionsArray = Array.isArray(predictions) ? predictions : []
  
  const scatterData = predictionsArray
    .filter(p => 
      p && 
      typeof p === 'object' &&
      p.PredictedSalary !== null && 
      p.PredictedSalary !== undefined &&
      p.ActualSalaryYearly !== null && 
      p.ActualSalaryYearly !== undefined &&
      !isNaN(parseFloat(p.PredictedSalary)) &&
      !isNaN(parseFloat(p.ActualSalaryYearly)) &&
      parseFloat(p.PredictedSalary) >= 10000 && 
      parseFloat(p.ActualSalaryYearly) >= 10000 &&
      parseFloat(p.PredictedSalary) <= 500000 &&
      parseFloat(p.ActualSalaryYearly) <= 500000
    )
    .slice(0, 1000)  // Limit to 1000 points for performance
    .map(p => {
      const predicted = parseFloat(p.PredictedSalary)
      const actual = parseFloat(p.ActualSalaryYearly)
      return {
        predicted: predicted,
        actual: actual,
        gap: actual - predicted,
        postingId: p.PostingID
      }
    })
  
  // Prepare line data for predicted and actual trends (sorted by predicted for smooth lines)
  const sortedData = scatterData.length > 0 ? [...scatterData].sort((a, b) => a.predicted - b.predicted) : []

  if (loading) {
    return <div className="loading">Loading predictions...</div>
  }

  // Show error message if no data loaded
  if (!kpis && !accuracy && gaps.length === 0 && predictions.length === 0) {
    return (
      <div className="predictions-page">
        <div className="page-header">
          <h1>Salary Predictions</h1>
        </div>
        <div style={{ padding: '20px', textAlign: 'center' }}>
          <p>Unable to load prediction data. Please check the backend connection.</p>
          <p style={{ fontSize: '0.9em', color: '#666' }}>Check browser console for errors.</p>
        </div>
      </div>
    )
  }

  // Check if any filters are active
  const hasActiveFilters = filters.industry || filters.experience_level || filters.compensation_type

  return (
    <div className="predictions-page">
      <div className="page-header">
        <h1>Salary Predictions</h1>
        {hasActiveFilters && (
          <p style={{ fontSize: '0.9em', color: '#666', marginTop: '5px' }}>
            Filters active: {filters.industry && `Industry: ${filters.industry} `}
            {filters.experience_level && `Level: ${filters.experience_level} `}
            {filters.compensation_type && `Type: ${filters.compensation_type}`}
          </p>
        )}
      </div>

      <FilterBar onFilterChange={setFilters} filters={filters} />

      {/* KPI Cards */}
      <div className="kpi-grid">
        <KPICard
          title="Median Salary"
          value={kpis?.median_salary || 0}
          subtitle="Middle value salary"
          color="blue"
          format="currency"
        />
        <KPICard
          title="Average Salary"
          value={kpis?.average_salary || 0}
          subtitle="Mean salary value"
          color="orange"
          format="currency"
        />
        <KPICard
          title="Predicted Salary"
          value={kpis?.predicted_salary || 0}
          subtitle="Average predicted salary"
          color="purple"
          format="currency"
        />
        <KPICard
          title="Highest Paying Industry"
          value={kpis?.highest_paying_industry || 'N/A'}
          subtitle={kpis?.highest_paying_industry_salary ? `$${Math.round(kpis.highest_paying_industry_salary).toLocaleString()}/year` : 'No data'}
          color="green"
          format="text"
        />
      </div>

      {/* Charts */}
      <div className="charts-grid">
        {/* Predicted vs Actual Scatter - Individual Predictions */}
        <div className="chart-card full-width">
          <h3>Predicted vs Actual Salary</h3>
          <p style={{ fontSize: '0.9em', color: '#666', marginTop: '-10px', marginBottom: '10px' }}>
            Each point represents one job posting's predicted vs actual salary. Data filtered by your current selections.
          </p>
          {sortedData.length === 0 ? (
            <div style={{ padding: '40px', textAlign: 'center', color: '#666' }}>
              No data available for the selected filters.
            </div>
          ) : (
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={sortedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                dataKey="predicted" 
                name="Predicted"
                label={{ value: 'Predicted Salary ($)', position: 'insideBottom', offset: -5 }}
                domain={['dataMin', 'dataMax']}
              />
              <YAxis 
                type="number" 
                name="Actual"
                label={{ value: 'Actual Salary ($)', angle: -90, position: 'insideLeft' }}
                domain={['dataMin', 'dataMax']}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (!active || !payload || !payload.length) return null;
                  
                  const data = payload[0]?.payload;
                  if (!data) return null;
                  
                  return (
                    <div style={{
                      backgroundColor: 'white',
                      border: '1px solid #ccc',
                      borderRadius: '4px',
                      padding: '10px',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                    }}>
                      {data.postingId && (
                        <div style={{ marginBottom: '8px', fontWeight: 'bold', color: '#333' }}>
                          Posting ID: {data.postingId}
                        </div>
                      )}
                      <div style={{ color: '#2ca02c', marginBottom: '4px' }}>
                        <strong>Predicted Salary:</strong> ${Math.round(data.predicted).toLocaleString()}
                      </div>
                      <div style={{ color: '#1f77b4', marginBottom: '4px' }}>
                        <strong>Actual Salary:</strong> ${Math.round(data.actual).toLocaleString()}
                      </div>
                      <div style={{ color: data.gap > 0 ? '#2ca02c' : '#d62728' }}>
                        <strong>Gap:</strong> ${Math.round(Math.abs(data.gap)).toLocaleString()} 
                        ({data.gap > 0 ? '+' : ''}{((data.gap / data.actual) * 100).toFixed(1)}%)
                      </div>
                    </div>
                  );
                }}
              />
              <Legend
                payload={[
                  { value: 'Predicted Salary', type: 'line', id: 'predicted', color: '#2ca02c' },
                  { value: 'Actual Salary', type: 'line', id: 'actual', color: '#1f77b4' }
                ]}
              />
              {/* Green line for Predicted trend */}
              <Line 
                type="monotone" 
                dataKey="predicted" 
                stroke="#2ca02c" 
                strokeWidth={2}
                dot={false}
                name="Predicted Salary"
              />
              {/* Blue line for Actual trend */}
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke="#1f77b4" 
                strokeWidth={2}
                dot={false}
                name="Actual Salary"
              />
              {/* Scatter points showing individual data points (not shown in legend to avoid confusion) */}
              <Scatter dataKey="actual" fill="#1f77b4">
                {sortedData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.gap > 0 ? '#2ca02c' : entry.gap < -10000 ? '#d62728' : '#1f77b4'} />
                ))}
              </Scatter>
            </ComposedChart>
          </ResponsiveContainer>
          )}
        </div>

        {/* Gap Categories */}
        <div className="chart-card">
          <h3>Prediction Gap Categories</h3>
          {categoryData.length === 0 ? (
            <div style={{ padding: '40px', textAlign: 'center', color: '#666' }}>
              No gap data available.
            </div>
          ) : (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#1f77b4">
                {categoryData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.name === 'Overpaying' ? '#2ca02c' : entry.name === 'Underpaying' ? '#d62728' : '#1f77b4'} 
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Top Gaps Table - Aggregated by Industry */}
      {/* 
        DATA SOURCE: transformed_predictions.csv
        - PredictedSalary: Average predicted salary per industry (avgPred)
        - SalaryMid: Average actual salary per industry (avgActual from ActualSalaryYearly)
        - Both in normalized yearly units
        - Aggregated by Industry with filters applied
      */}
      <div className="table-card">
        <h3>Top Prediction Gaps by Industry</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Industry</th>
                <th>Experience Level</th>
                <th>Predicted</th>
                <th>Actual</th>
                <th>Gap</th>
                <th>Gap %</th>
                <th>Category</th>
              </tr>
            </thead>
            <tbody>
              {gaps
                .slice(0, 10)  // Already sorted by absolute Gap % descending from backend
                .map((gap, idx) => (
                  <tr key={idx}>
                    <td>{gap.Industry || 'Unknown'}</td>
                    <td>{gap.RoleLevel || 'N/A'}</td>
                    <td>${Math.round(gap.PredictedSalary).toLocaleString()}</td>
                    <td>${Math.round(gap.SalaryMid).toLocaleString()}</td>
                    <td className={gap.Gap > 0 ? 'positive' : 'negative'}>
                      ${Math.round(Math.abs(gap.Gap)).toLocaleString()}
                    </td>
                    <td>{gap.GapPct.toFixed(1)}%</td>
                    <td>
                      <span className={`badge badge-${gap.Category.toLowerCase()}`}>
                        {gap.Category}
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default Predictions


