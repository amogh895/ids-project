import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
CORS(app)

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
# PREDICT API (DUMMY LOGIC)
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        duration = float(data["duration"])
        src_bytes = int(data["src_bytes"])
        dst_bytes = int(data["dst_bytes"])

        # 🔥 SMART DUMMY LOGIC (FOR DEMO)
        if src_bytes > 5000 or dst_bytes < 50 or duration > 100:
            result = "Attack"
        else:
            result = "Normal"

        return jsonify({
            "prediction": result
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