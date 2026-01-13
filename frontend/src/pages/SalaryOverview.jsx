import React, { useState, useEffect } from 'react'
import FilterBar from '../components/FilterBar'
import KPICard from '../components/KPICard'
import { getSalaryByIndustry, getSalaryByExperienceLevel, getTopSkills, getSalaryInsightsKPIs } from '../api/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts'
import './SalaryOverview.css'

function SalaryOverview() {
  const [filters, setFilters] = useState({})
  const [kpis, setKpis] = useState(null)
  const [salaryByIndustry, setSalaryByIndustry] = useState([])
  const [salaryByExperienceLevel, setSalaryByExperienceLevel] = useState([])
  const [topSkills, setTopSkills] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [filters])

  const loadData = async () => {
    setLoading(true)
    try {
      const [kpisRes, industryRes, experienceLevelRes, skillsRes] = await Promise.all([
        getSalaryInsightsKPIs(filters),
        getSalaryByIndustry(filters),
        getSalaryByExperienceLevel(filters),
        getTopSkills(filters)
      ])
      
      setKpis(kpisRes.data)
      setSalaryByIndustry(industryRes.data)
      setSalaryByExperienceLevel(experienceLevelRes.data)
      setTopSkills(skillsRes.data)
    } catch (error) {
      console.error('Error loading data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="loading">Loading salary data...</div>
  }

  return (
    <div className="salary-overview-page">
      <div className="page-header">
        <h1>Salary Insights</h1>
        <p>Market Salary Insights by Industry, Experience Level, and Skills</p>
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
          title="Highest Paying Industry"
          value={kpis?.highest_paying_industry || 'N/A'}
          subtitle={kpis?.highest_paying_industry_salary ? `$${Math.round(kpis.highest_paying_industry_salary).toLocaleString()}/year` : 'No data'}
          color="green"
          format="text"
        />
        <KPICard
          title="Highest Paying Skill"
          value={kpis?.highest_paying_skill || 'N/A'}
          subtitle={kpis?.highest_paying_skill_salary ? `$${Math.round(kpis.highest_paying_skill_salary).toLocaleString()}/year` : 'No data'}
          color="purple"
          format="text"
        />
      </div>

      <div className="charts-grid">
        {/* Salary by Industry */}
        <div className="chart-card full-width">
          <h3>Average Salary by Industry</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={salaryByIndustry.slice(0, 15)} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis 
                dataKey="Industry" 
                type="category" 
                width={200}
                tick={{ fontSize: 12 }}
              />
              <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
              <Legend />
              <Bar dataKey="average" fill="#1f77b4" name="Average" />
              <Bar dataKey="median" fill="#2ca02c" name="Median" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Salary by Experience Level */}
        <div className="chart-card">
          <h3>Average Salary by Experience Level</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={salaryByExperienceLevel}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="RoleLevel" />
              <YAxis />
              <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
              <Legend />
              <Bar dataKey="average" fill="#1f77b4" name="Average" />
              <Bar dataKey="median" fill="#2ca02c" name="Median" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Top Skills by Salary Impact */}
        <div className="chart-card">
          <h3>Top Skills by Average Salary</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topSkills.slice(0, 10)} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="skill" type="category" width={120} />
              <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
              <Bar dataKey="average_salary" fill="#9467bd" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

export default SalaryOverview



