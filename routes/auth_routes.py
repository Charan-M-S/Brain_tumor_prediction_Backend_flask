from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

auth_bp = Blueprint("auth", __name__)
user_model = None  # will be injected

# Setter function
def set_user_model(model):
    global user_model
    user_model = model

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    if user_model.find_user(email):  # ✅ use the injected instance
        return jsonify({"error": "User already exists"}), 400

    user_model.create_user(
        name=data.get("name"),
        email=data.get("email"),
        password=data.get("password"),
        role=data.get("role")
    )
    return jsonify({"message": "User registered successfully"}), 201

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = user_model.verify_password(email, password)
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    access_token = create_access_token(
        identity=str(user["_id"]),
        additional_claims={"role": user["role"]},
        expires_delta=datetime.timedelta(hours=8)
    )

    return jsonify({
        "token": access_token,
        "id": str(user["_id"]),
        "role": user["role"],
        "name": user["name"]
    }), 200
