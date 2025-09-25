import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from flask import Flask, jsonify
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager
from config import MONGO_URI, SECRET_KEY
from utils.model_inference import load_brain_model
from flask_cors import CORS


# Routes
from routes.auth_routes import auth_bp, set_user_model
from routes.admin_routes import admin_bp
from routes.doctor_routes import doctor_bp
from routes.patient_routes import patient_bp

from models.user_model import User
from models.prediction_model import Prediction

app = Flask(__name__)
CORS(app)
# Config
app.config["MONGO_URI"] = MONGO_URI
app.config["JWT_SECRET_KEY"] = SECRET_KEY

# Initialize Mongo
mongo = PyMongo(app)

# Initialize models
user_model = User(mongo)
set_user_model(user_model)  # inject before registering blueprint

prediction_model = Prediction(mongo)

# Inject models into routes
def inject_models():
    auth_bp.user_model = user_model
    admin_bp.user_model = user_model
    doctor_bp.user_model = user_model
    doctor_bp.prediction_model = prediction_model
    patient_bp.user_model = user_model
    patient_bp.prediction_model = prediction_model
inject_models()

# Initialize JWT
jwt = JWTManager(app)

# Register Blueprints
app.register_blueprint(auth_bp, url_prefix="/auth")
app.register_blueprint(admin_bp, url_prefix="/admin")
app.register_blueprint(doctor_bp, url_prefix="/doctor")
app.register_blueprint(patient_bp, url_prefix="/patient")

@app.route("/")
def home():
    return jsonify({"message": "Brain Tumor MVP Backend Running"})

if __name__ == "__main__":
    try:
        load_brain_model()
    except Exception as e:
        print(f"⚠️ Model failed to load: {e}")
    os.makedirs("uploads/mri_images", exist_ok=True)
    os.makedirs("uploads/reports", exist_ok=True)
    app.run(debug=True, use_reloader=False)
