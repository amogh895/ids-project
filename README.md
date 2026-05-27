#  Intrusion Detection System (IDS)

##  Project Overview
This project is a web-based Intrusion Detection System designed to identify whether network traffic is Normal or represents a potential Attack.  
It uses a simple intelligent logic (can be extended to machine learning) to analyze key network parameters and provide instant results.

---

##  Objective
The main goal of this project is to:
- Detect suspicious network behavior
- Classify traffic into Normal or Attack
- Provide a simple and interactive interface for users
- Demonstrate the working of an IDS system

---

## Features
- Login system (User & Admin)
- Input network parameters:
  - Duration
  - Source Bytes
  - Destination Bytes
-  Instant prediction (Normal / Attack)
-  Alert message for attacks
-  Admin panel (UI-based controls)

---

## Technologies Used
- **Frontend:** ReactJS (HTML, CSS, JavaScript)
- **Backend:** Flask (Python)
- **Logic:** Rule-based / ML-ready structure
- **Tools:** VS Code, GitHub

---

## Screenshots

### Login Page
![Login](screenshots/login.png)

###  User Dashboard
![Dashboard](screenshots/UserDashboard.png)

### Normal Traffic Output
![Normal](screenshots/Prediction(Normal).png)

###  Attack Detection
![Attack](screenshots/Prediction(Threat).png)

###  Admin Panel
![Admin](screenshots/AdminDashboard.png)

---

## How to Run the Project

###  Backend
```bash
cd backend
python app.py

###  Frontend
cd frontend
npm install
npm run dev

## Live Demo

[Click Here to Open Project] (https://ids-project-delta.vercel.app)
