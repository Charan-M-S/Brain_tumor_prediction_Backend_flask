from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os, uuid
from utils.model_inference import predict_mri, generate_pdf_report

doctor_bp = Blueprint("doctor", __name__)
doctor_bp.prediction_model = None  # injected from app.py
doctor_bp.user_model = None        # injected from app.py

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ["jpg","jpeg","png"]

@doctor_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict_route():
    try:
        doctor_id = get_jwt_identity()
        patient_id = request.form.get("patient_id")

        if not patient_id:
            return jsonify({"error": "patient_id required"}), 400

        # Check file
        file = request.files.get("mri_image")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads/mri_images", f"{uuid.uuid4().hex}_{filename}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        # Predict
        predicted_class, confidence = predict_mri(file_path)

        # Generate PDF
        patient = doctor_bp.user_model.find_user_by_id(patient_id)
        doctor = doctor_bp.user_model.find_user_by_id(doctor_id)
        report_path = os.path.join("uploads/reports", f"{uuid.uuid4().hex}_report.pdf")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        generate_pdf_report(
            patient_name=patient.get("name", "Unknown"),
            patient_id=patient_id,
            doctor_name=doctor.get("name", "Unknown"),
            predicted_class=predicted_class,
            confidence=confidence,
            notes="",
            save_path=report_path
        )

        # Store prediction in DB
        pred_id = doctor_bp.prediction_model.create_prediction(
            patient_id=patient_id,
            doctor_id=doctor_id,
            image_path=file_path,
            predicted_class=predicted_class,
            confidence=confidence,
            validated=False,
            notes="",
            report_pdf_path=report_path
        )

        return jsonify({
            "message": "Prediction created",
            "prediction_id": pred_id,
            "class": predicted_class,
            "confidence": confidence
        }), 201

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@doctor_bp.route("/validate/<prediction_id>", methods=["POST"])
@jwt_required()
def validate_prediction(prediction_id):
    data = request.json
    validated = data.get("validated", True)
    notes = data.get("notes", "")
    success = doctor_bp.prediction_model.update_prediction(prediction_id, validated, notes)
    if success:
        return jsonify({"message": "Prediction updated successfully"})
    else:
        return jsonify({"error": "Prediction not found"}), 404


@doctor_bp.route("/predictions", methods=["GET"])
@jwt_required()
def my_predictions():
    doctor_id = get_jwt_identity()
    preds = doctor_bp.prediction_model.get_predictions_by_doctor(doctor_id)
    result = []
    for p in preds:
        result.append({
            "_id": str(p["_id"]),
            "patient_id": str(p["patient_id"]),
            "doctor_id": str(p["doctor_id"]),
            "mri_image_path": p.get("mri_image_path", ""),
            "predicted_class": p.get("predicted_class", ""),
            "confidence": p.get("confidence", 0.0),
            "validated": p.get("validated", False),
            "notes": p.get("notes", ""),
            "report_pdf_path": p.get("report_pdf_path", ""),
            "status": p.get("status", "pending"),
            "created_at": p.get("created_at").isoformat() if p.get("created_at") else None
        })
    print(jsonify(result))
    return jsonify(result)


@doctor_bp.route("/report/<prediction_id>", methods=["GET"])
@jwt_required()
def get_report(prediction_id):
    pred = doctor_bp.prediction_model.get_prediction_by_id(prediction_id)
    if not pred or not pred.get("report_pdf_path"):
        return jsonify({"error": "Report not found"}), 404
    return send_file(pred["report_pdf_path"], as_attachment=True)
