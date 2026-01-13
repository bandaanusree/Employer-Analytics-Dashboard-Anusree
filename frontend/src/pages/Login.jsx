import React, { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import './Login.css'

function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  // Redirect if already authenticated
  useEffect(() => {
    const authToken = localStorage.getItem('authToken')
    if (authToken) {
      navigate('/', { replace: true })
    }
  }, [navigate])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    // Simple authentication (for demo purposes)
    // In production, this would call an API
    if (email && password) {
      // Simulate API call
      setTimeout(() => {
        // Store auth token in localStorage
        localStorage.setItem('authToken', 'demo-token')
        localStorage.setItem('userEmail', email)
        setLoading(false)
        navigate('/')
      }, 500)
    } else {
      setError('Please enter both email and password')
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h1>Employability Analytics Dashboard</h1>
          <p>Sign in to access your salary and compensation insights</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {error && <div className="error-message">{error}</div>}

          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
              required
              autoFocus
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
            />
          </div>

          <button type="submit" className="login-button" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <div className="login-footer">
          <p>
            Don't have an account? <Link to="/signup" className="footer-link">Create Account</Link>
          </p>
        </div>
      </div>
    </div>
  )
}

export default Login


