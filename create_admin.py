from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash
from config import MONGO_URI
from flask import Flask

# Temporary app just to access mongo
app = Flask(__name__)
app.config["MONGO_URI"] = MONGO_URI
mongo = PyMongo(app)

with app.app_context():
    username = "admin1"
    password = "admin123"
    role = "admin"

    existing = mongo.db.users.find_one({"username": username})
    if existing:
        print("⚠️ Admin already exists, skipping.")
    else:
        mongo.db.users.insert_one({
            "username": username,
            "password_hash": generate_password_hash(password),
            "role": role
        })
        print(f"✅ Admin user '{username}' created with password '{password}'")
