import React from 'react'
import './KPICard.css'

function KPICard({ title, value, subtitle, trend, color = 'blue', format = 'auto' }) {
  const formatValue = (val) => {
    if (format === 'text') {
      return val
    }
    if (typeof val === 'number') {
      if (format === 'currency') {
        return `$${val.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
      } else if (format === 'number') {
        return val.toLocaleString()
      } else {
        // Auto format
        if (val >= 1000000) {
          return `$${(val / 1000000).toFixed(1)}M`
        } else if (val >= 1000) {
          return `$${(val / 1000).toFixed(0)}K`
        } else {
          return `$${val.toFixed(0)}`
        }
      }
    }
    return val
  }

  return (
    <div className={`kpi-card kpi-card-${color}`}>
      <div className="kpi-header">
        <div className="kpi-title">{title}</div>
      </div>
      <div className="kpi-value">{formatValue(value)}</div>
      {subtitle && <div className="kpi-subtitle">{subtitle}</div>}
      {trend && (
        <div className={`kpi-trend kpi-trend-${trend.type}`}>
          {trend.type === 'up' ? '↑' : trend.type === 'down' ? '↓' : '→'} {trend.value}
        </div>
      )}
    </div>
  )
}

export default KPICard
