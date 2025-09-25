from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt
from bson import ObjectId
from werkzeug.security import generate_password_hash

admin_bp = Blueprint("admin", __name__)
user_model = None  # injected from app.py

def is_admin():
    claims = get_jwt()
    return claims.get("role") == "admin"

@admin_bp.route("/create_user", methods=["POST"])
@jwt_required()
def create_user():
    if not is_admin():
        return jsonify({"error": "Admin only"}), 403

    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role")
    assigned_doctor_id = data.get("assigned_doctor_id")

    user_id = user_model.create_user(name, email, password, role, assigned_doctor_id)
    if not user_id:
        return jsonify({"error": "User creation failed"}), 400

    return jsonify({"message": f"{role.capitalize()} created successfully", "user_id": user_id}), 201

@admin_bp.route("/list_users", methods=["GET"])
@jwt_required()
def list_users():
    if not is_admin():
        return jsonify({"error": "Admin only"}), 403
    users = list(user_model.collection.find({}, {"password_hash": 0}))
    for u in users:
        u["_id"] = str(u["_id"])
        if "assigned_doctor_id" in u:
            u["assigned_doctor_id"] = str(u["assigned_doctor_id"])
    return jsonify(users), 200

@admin_bp.route("/reset_password", methods=["POST"])
@jwt_required()
def reset_password():
    if not is_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.json
    user_id = data.get("user_id")
    new_password = data.get("new_password")
    if not user_id or not new_password:
        return jsonify({"error": "user_id and new_password required"}), 400
    success = user_model.reset_password(user_id, new_password)
    if success:
        return jsonify({"message": "Password reset successfully"})
    else:
        return jsonify({"error": "Failed to reset password"}), 400
