import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
from collections import deque

app = Flask(__name__)
CORS(app)

# Rolling window for source bytes
recent_src_bytes = deque(maxlen=10)

# =========================
# DATABASE CONFIGURATION
# =========================
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =========================
# MODELS
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="user") # 'admin' or 'user'
    status = db.Column(db.String(50), nullable=False, default="active")

# Create tables and default users if not exists
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        default_admin = User(username="admin", password="admin123", role="admin", status="active")
        db.session.add(default_admin)
        db.session.commit()
    if not User.query.filter_by(username="user").first():
        default_user = User(username="user", password="1234", role="user", status="active")
        db.session.add(default_user)
        db.session.commit()

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return "IDS Backend Running 🚀"

# =========================
# LOGIN API
# =========================
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        user = User.query.filter_by(username=username, password=password).first()
        if user:
            return jsonify({"success": True, "role": user.role, "username": user.username})
        else:
            return jsonify({"success": False, "error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# USER MANAGEMENT APIs
# =========================
@app.route("/api/users", methods=["GET"])
def get_users():
    users = User.query.all()
    users_list = []
    for u in users:
        users_list.append({
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "status": u.status
        })
    return jsonify(users_list)

@app.route("/api/users", methods=["POST"])
def create_user():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        role = data.get("role", "user")

        if not username or not password:
            return jsonify({"success": False, "error": "Username and password required"}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({"success": False, "error": "Username already exists"}), 400

        new_user = User(username=username, password=password, role=role, status="active")
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"success": True, "message": "User created successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
            
        if user.username == "admin":
            return jsonify({"success": False, "error": "Cannot delete default admin"}), 400

        db.session.delete(user)
        db.session.commit()
        return jsonify({"success": True, "message": "User deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# PREDICT API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        duration = float(data["duration"])
        src_bytes = int(data["src_bytes"])
        dst_bytes = int(data["dst_bytes"])
        count = int(data.get("count", 0))
        srv_count = int(data.get("srv_count", 0))

        # Update rolling window
        recent_src_bytes.append(src_bytes)
        
        # Calculate rolling mean and std
        src_bytes_roll_mean = float(np.mean(recent_src_bytes))
        src_bytes_roll_std = float(np.std(recent_src_bytes)) if len(recent_src_bytes) > 1 else 0.0
        
        # Calculate trend (difference from previous packet)
        src_bytes_trend = float(recent_src_bytes[-1] - recent_src_bytes[-2]) if len(recent_src_bytes) > 1 else 0.0

        # Try to load the trained model
        model_path = os.path.join(basedir, "models", "model.pkl")
        
        feature_names = ["Duration", "Src Bytes", "Dst Bytes", "Host Count", "Srv Count", "Roll Mean", "Roll Std", "Trend"]
        feature_values = [duration, src_bytes, dst_bytes, count, srv_count, src_bytes_roll_mean, src_bytes_roll_std, src_bytes_trend]

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            features_arr = np.array([feature_values])
            prediction = model.predict(features_arr)
            result = "Attack" if prediction[0] == 1 else "Normal"
        else:
            # 🔥 SMART DUMMY LOGIC (FOR DEMO)
            if src_bytes > 5000 or dst_bytes < 50 or duration > 100 or count > 50 or srv_count > 50:
                result = "Attack"
            else:
                result = "Normal"

        return jsonify({
            "prediction": result,
            "feature_names": feature_names,
            "feature_values": feature_values
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))