import React, { useState, useEffect } from 'react'
import { getIndustries, getExperienceLevels, getCompensationTypes } from '../api/api'
import './FilterBar.css'

function FilterBar({ onFilterChange, filters = {} }) {
  const [localFilters, setLocalFilters] = useState({
    industry: filters.industry || '',
    experience_level: filters.experience_level || '',
    compensation_type: filters.compensation_type || ''
  })
  
  const [options, setOptions] = useState({
    industries: [],
    experienceLevels: [],
    compensationTypes: []
  })

  useEffect(() => {
    // Load filter options
    Promise.all([
      getIndustries(),
      getExperienceLevels(),
      getCompensationTypes()
    ]).then(([industriesRes, experienceLevelsRes, compTypesRes]) => {
      setOptions({
        industries: industriesRes.data,
        experienceLevels: experienceLevelsRes.data,
        compensationTypes: compTypesRes.data
      })
    })
  }, [])

  const handleFilterChange = (key, value) => {
    const newFilters = { ...localFilters, [key]: value }
    setLocalFilters(newFilters)
    onFilterChange(newFilters)
  }

  const clearFilters = () => {
    const cleared = { industry: '', experience_level: '', compensation_type: '' }
    setLocalFilters(cleared)
    onFilterChange(cleared)
  }

  return (
    <div className="filter-bar">
      <div className="filter-group">
        <label>Industry</label>
        <select
          value={localFilters.industry}
          onChange={(e) => handleFilterChange('industry', e.target.value)}
        >
          <option value="">All Industries</option>
          {options.industries.map(industry => (
            <option key={industry} value={industry}>{industry}</option>
          ))}
        </select>
      </div>
      
      <div className="filter-group">
        <label>Experience Level</label>
        <select
          value={localFilters.experience_level}
          onChange={(e) => handleFilterChange('experience_level', e.target.value)}
        >
          <option value="">All Levels</option>
          {options.experienceLevels.map(level => (
            <option key={level} value={level}>{level}</option>
          ))}
        </select>
      </div>
      
      <div className="filter-group">
        <label>Compensation Type</label>
        <select
          value={localFilters.compensation_type}
          onChange={(e) => handleFilterChange('compensation_type', e.target.value)}
        >
          <option value="">All Types</option>
          {options.compensationTypes.map(type => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </div>
      
      <button className="clear-filters" onClick={clearFilters}>
        Clear Filters
      </button>
    </div>
  )
}

export default FilterBar



