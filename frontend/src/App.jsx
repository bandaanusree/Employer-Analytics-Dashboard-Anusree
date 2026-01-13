import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate, useNavigate } from 'react-router-dom'
import Home from './pages/Home'
import SalaryOverview from './pages/SalaryOverview'
import Predictions from './pages/Predictions'
import Login from './pages/Login'
import Signup from './pages/Signup'
import './App.css'

function Navigation() {
  const location = useLocation()
  const navigate = useNavigate()
  const [userEmail, setUserEmail] = useState('')

  useEffect(() => {
    const email = localStorage.getItem('userEmail')
    if (email) {
      setUserEmail(email)
    }
  }, [])

  const handleLogout = () => {
    localStorage.removeItem('authToken')
    localStorage.removeItem('userEmail')
    navigate('/login', { replace: true })
  }
  
  const navItems = [
    { path: '/', label: 'Home', icon: '🏠' },
    { path: '/salary-overview', label: 'Salary Insights', icon: '📊' },
    { path: '/predictions', label: 'Predictions', icon: '🔮' }
  ]
  
  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="nav-brand">
          <h1>Employability Analytics Dashboard</h1>
          <span className="nav-subtitle">Salary & Compensation Analytics</span>
        </div>
        <div className="nav-right">
          <div className="nav-links">
            {navItems.map(item => (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
              >
                <span className="nav-icon">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>
          {userEmail && (
            <div className="nav-user">
              <span className="user-email">{userEmail}</span>
              <button onClick={handleLogout} className="logout-button">
                Logout
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  )
}

function ProtectedRoute({ children }) {
  const isAuthenticated = localStorage.getItem('authToken')
  // Temporarily allow access for debugging - remove auth check
  // return isAuthenticated ? children : <Navigate to="/login" replace />
  return children
}

function App() {
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <Navigation />
                <main className="main-content">
                  <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/salary-overview" element={<SalaryOverview />} />
                    <Route path="/predictions" element={<Predictions />} />
                  </Routes>
                </main>
              </ProtectedRoute>
            }
          />
        </Routes>
      </div>
    </Router>
  )
}

export default App
