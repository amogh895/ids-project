import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

// --- SVG Icons ---
const ShieldIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>;
const UserIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>;
const LogOutIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>;
const AlertTriangleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>;
const CheckCircleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>;
const ActivityIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>;
const UsersIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>;
const TrashIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>;

const API_BASE = "https://ids-project-ohnh.onrender.com" ;

function App() {
  // LOGIN STATE
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [role, setRole] = useState("");
  const [loginData, setLoginData] = useState({ username: "", password: "" });
  const [loginError, setLoginError] = useState("");
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  // USER INPUT STATE
  const [formData, setFormData] = useState({ duration: "", src_bytes: "", dst_bytes: "", count: "", srv_count: "" });
  const [result, setResult] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [logs, setLogs] = useState([]);

  // SECURITY STATS CALCULATION
  const totalPackets = logs.length;
  const totalNormal = logs.filter(l => l.prediction === "Normal").length;
  const totalAttacks = logs.filter(l => l.prediction !== "Normal" && !l.prediction.startsWith("Error")).length;
  
  let systemStatus = "MONITORING";
  let systemStatusColor = "var(--text-muted)";
  if (logs.length > 0) {
    if (logs[0].prediction !== "Normal" && !logs[0].prediction.startsWith("Error")) {
      systemStatus = "THREAT ALERT";
      systemStatusColor = "var(--danger)";
    } else if (logs[0].prediction === "Normal") {
      systemStatus = "SECURE";
      systemStatusColor = "var(--success)";
    }
  }
  
  // USER MANAGEMENT STATE
  const [users, setUsers] = useState([]);
  const [newUser, setNewUser] = useState({ username: "", password: "", role: "user" });
  const [userMsg, setUserMsg] = useState({ text: "", type: "" });

  // FETCH USERS EFFECT
  useEffect(() => {
    if (isLoggedIn && role === "admin") {
      fetchUsers();
    }
  }, [isLoggedIn, role]);

  const fetchUsers = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/users`);
      setUsers(res.data);
    } catch (err) {
      console.error("Failed to fetch users");
    }
  };

  // LOGIN HANDLER
  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError("");
    setIsLoggingIn(true);
    
    try {
      const res = await axios.post(`${API_BASE}/login`, {
        username: loginData.username,
        password: loginData.password
      });
      
      if (res.data.success) {
        setRole(res.data.role);
        setIsLoggedIn(true);
      }
    } catch (err) {
      setLoginError(err.response?.data?.error || "Connection error. Is the backend running?");
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setRole("");
    setLoginData({ username: "", password: "" });
    setResult("");
    setFormData({ duration: "", src_bytes: "", dst_bytes: "", count: "", srv_count: "" });
  };

  // USER INPUT HANDLER
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // PREDICTION
  const handlePredict = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult("");

    try {
      const res = await axios.post(`${API_BASE}/predict`, {
        duration: Number(formData.duration),
        src_bytes: Number(formData.src_bytes),
        dst_bytes: Number(formData.dst_bytes),
        count: Number(formData.count),
        srv_count: Number(formData.srv_count)
      });

      if (res.data.error) {
        throw new Error(res.data.error);
      }

      const prediction = res.data.prediction;
      
      setTimeout(() => {
        setResult(prediction);
        setIsLoading(false);
        const newLog = { time: new Date().toLocaleTimeString(), prediction };
        setLogs([newLog, ...logs]);
      }, 600);
    } catch (err) {
      const errMsg = err.message || "Connection error. Unable to reach prediction engine.";
      setTimeout(() => {
        setResult("Error: " + errMsg);
        setIsLoading(false);
      }, 600);
    }
  };

  // ADMIN FUNCTIONS

  const handleCreateUser = async (e) => {
    e.preventDefault();
    setUserMsg({ text: "", type: "" });
    try {
      const res = await axios.post(`${API_BASE}/api/users`, newUser);
      if (res.data.success) {
        setUserMsg({ text: res.data.message, type: "success" });
        setNewUser({ username: "", password: "", role: "user" });
        fetchUsers();
      }
    } catch (err) {
      setUserMsg({ text: err.response?.data?.error || "Error creating user", type: "error" });
    }
  };

  const handleDeleteUser = async (id) => {
    try {
      const res = await axios.delete(`${API_BASE}/api/users/${id}`);
      if (res.data.success) {
        setUserMsg({ text: res.data.message, type: "success" });
        fetchUsers();
      }
    } catch (err) {
      setUserMsg({ text: err.response?.data?.error || "Error deleting user", type: "error" });
    }
  };

  // LOGIN PAGE
  if (!isLoggedIn) {
    return (
      <div className="app-wrapper">
        <div className="auth-container fade-in">
          <div className="logo-section">
            <ShieldIcon />
            <h2>SmartIDS</h2>
            <p>Advanced Threat Detection</p>
            <br></br>
            <h4>Login</h4>
          </div>
          
          <form className="auth-form" onSubmit={handleLogin}>
            <div className="input-group">
              <label>Username</label>
              <input type="text" required placeholder="Enter your username" value={loginData.username} onChange={(e) => setLoginData({ ...loginData, username: e.target.value })} />
            </div>

            <div className="input-group">
              <label>Password</label>
              <input type="password" required placeholder="Enter your password" value={loginData.password} onChange={(e) => setLoginData({ ...loginData, password: e.target.value })} />
            </div>

            {loginError && <div className="error-toast">{loginError}</div>}

            <button type="submit" className="primary-btn pulse-hover" disabled={isLoggingIn}>
              {isLoggingIn ? <span className="spinner"></span> : "Secure Login"}
            </button>

            
          </form>
        </div>
      </div>
    );
  }

  // DASHBOARD PAGE
  return (
    <div className="app-wrapper dashboard-layout fade-in">
      <nav className="navbar">
        <div className="nav-brand">
          <ShieldIcon />
          <span>SmartIDS</span>
        </div>
        <div className="nav-controls">
          <div className="user-badge">
            <UserIcon />
            <span>{role.toUpperCase()}</span>
          </div>
          <button onClick={handleLogout} className="icon-btn logout-btn">
            <LogOutIcon />
            <span>Logout</span>
          </button>
        </div>
      </nav>

      <main className="dashboard-content">
        
        {/* USER PANEL */}
        {role === "user" && (
          <div className="panel slide-up">
            <div className="panel-header">
              <ActivityIcon />
              <h2>Traffic Analysis</h2>
            </div>
            <p className="panel-desc">Analyze network packets for potential intrusions.</p>

            <form onSubmit={handlePredict} className="prediction-form">
              <div className="input-row">
                <div className="input-group">
                  <label>Duration (s)</label>
                  <input type="number" step="any" name="duration" value={formData.duration} placeholder="e.g. 1.5" onChange={handleChange} required />
                </div>
                <div className="input-group">
                  <label>Source Bytes</label>
                  <input type="number" name="src_bytes" value={formData.src_bytes} placeholder="e.g. 5400" onChange={handleChange} required />
                </div>
                <div className="input-group">
                  <label>Destination Bytes</label>
                  <input type="number" name="dst_bytes" value={formData.dst_bytes} placeholder="e.g. 120" onChange={handleChange} required />
                </div>
              </div>
              <div className="input-row">
                <div className="input-group">
                  <label>Host Conn Count (2s)</label>
                  <input type="number" name="count" value={formData.count} placeholder="e.g. 8" onChange={handleChange} required />
                </div>
                <div className="input-group">
                  <label>Service Conn Count (2s)</label>
                  <input type="number" name="srv_count" value={formData.srv_count} placeholder="e.g. 8" onChange={handleChange} required />
                </div>
              </div>

              <button type="submit" className="primary-btn analyze-btn" disabled={isLoading}>
                {isLoading ? <span className="spinner"></span> : "Run Analysis"}
              </button>
            </form>

            {result && (
              <div className={`result-card scale-in ${result === 'Normal' ? 'normal-theme' : result.startsWith('Error') ? 'error-theme' : 'attack-theme'}`}>
                {result === "Normal" && <><CheckCircleIcon /><div><h3>Traffic Normal</h3><p>No threats detected in this packet.</p></div></>}
                {result.startsWith("Error") && <><AlertTriangleIcon /><div><h3>Analysis Failed</h3><p>{result === "Error" ? "Unable to reach the prediction engine." : result.replace("Error: ", "")}</p></div></>}
                {result !== "Normal" && !result.startsWith("Error") && (
                  <><AlertTriangleIcon /><div><h3>Threat Detected: {result}</h3><p>Anomalous network activity matching attack signature flagged.</p></div></>
                )}
              </div>
            )}


          </div>
        )}

        {/* ADMIN PANEL */}
        {role === "admin" && (
          <div className="admin-grid slide-up">
            {/* Left Column: Management */}
            <div className="admin-col">
              <div className="panel animate-slide-up">
                <div className="panel-header">
                  <ActivityIcon />
                  <h2>Security Overview</h2>
                </div>
                <p className="panel-desc">Real-time statistics of analyzed traffic logs.</p>
                
                <div className="admin-section">
                  <div className="security-stats-grid">
                    <div className="stat-card">
                      <span className="stat-label">System Status</span>
                      <span className="stat-value" style={{ color: systemStatusColor }}>
                        {systemStatus}
                      </span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-label">Traffic Scanned</span>
                      <span className="stat-value">{totalPackets}</span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-label">Intrusions Blocked</span>
                      <span className="stat-value" style={{ color: totalAttacks > 0 ? "var(--danger)" : "inherit" }}>
                        {totalAttacks}
                      </span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-label">Normal Traffic</span>
                      <span className="stat-value" style={{ color: totalNormal > 0 ? "var(--success)" : "inherit" }}>
                        {totalNormal}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="panel">
                <div className="panel-header">
                  <UsersIcon />
                  <h2>User Management</h2>
                </div>
                <div className="admin-section">
                  <form className="create-user-form" onSubmit={handleCreateUser}>
                    <div className="input-group">
                      <input type="text" placeholder="New Username" required value={newUser.username} onChange={(e) => setNewUser({...newUser, username: e.target.value})} />
                    </div>
                    <div className="input-group">
                      <input type="password" placeholder="Password" required value={newUser.password} onChange={(e) => setNewUser({...newUser, password: e.target.value})} />
                    </div>
                    <div className="input-group">
                      <select value={newUser.role} onChange={(e) => setNewUser({...newUser, role: e.target.value})} className="role-select">
                        <option value="user">User</option>
                        <option value="admin">Admin</option>
                      </select>
                    </div>
                    <button type="submit" className="secondary-btn">Add User</button>
                  </form>
                  {userMsg.text && <div className={`status-badge ${userMsg.type === 'error' ? 'error' : ''}`} style={{marginTop: '10px'}}>{userMsg.text}</div>}

                  <div className="table-container" style={{marginTop: '20px'}}>
                    <table className="logs-table">
                      <thead>
                        <tr>
                          <th>Username</th>
                          <th>Role</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {users.map(u => (
                          <tr key={u.id} className="fade-in">
                            <td>{u.username}</td>
                            <td><span className={`status-pill ${u.role === 'admin' ? 'pill-admin' : 'pill-user'}`}>{u.role}</span></td>
                            <td>
                              {u.username !== "admin" && (
                                <button className="icon-btn delete-btn" onClick={() => handleDeleteUser(u.id)} title="Delete user">
                                  <TrashIcon />
                                </button>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Logs */}
            <div className="admin-col">
              <div className="panel h-100">
                <div className="panel-header">
                  <ActivityIcon />
                  <h2>Log Management</h2>
                </div>
                <div className="table-container">
                  <table className="logs-table">
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Prediction Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {logs.length === 0 ? (
                        <tr><td colSpan="2" className="empty-state">No logs available yet.</td></tr>
                      ) : (
                        logs.map((log, i) => (
                          <tr key={i} className="fade-in">
                            <td className="log-time">{log.time}</td>
                            <td>
                              <span className={`status-pill ${log.prediction === "Normal" ? "pill-normal" : log.prediction.startsWith("Error") ? "pill-user" : "pill-attack"}`}>
                                {log.prediction}
                              </span>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;