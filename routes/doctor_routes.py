from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os, uuid
from utils.model_inference import generate_pdf_report,predict_and_segment
from models.user_model import User 
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
        email = request.form.get("patient_email")
        patient_id = doctor_bp.user_model.find_patient_id_by_email(email=email)
        if not patient_id:
            return jsonify({"error": "Enter correct patient email"}), 400

        file = request.files.get("mri_image")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads/mri_images", f"{uuid.uuid4().hex}_{filename}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        # --- Run prediction + segmentation if needed --- #
        predicted_class, confidence, segmentation_path = predict_and_segment(file_path)

        # Store prediction in DB
        pred_id = doctor_bp.prediction_model.create_prediction(
            patient_id=patient_id,
            doctor_id=doctor_id,
            image_path=file_path,
            predicted_class=predicted_class,
            confidence=confidence,
            validated=False,
            notes="",
            report_pdf_path=None,  # No PDF for now
            segmentation_path=segmentation_path  # optional: add column in DB
        )

        response = {
            "message": "Prediction created",
            "prediction_id": pred_id,
            "class": predicted_class,
            "confidence": confidence
        }

        if segmentation_path:
            segmentation_web_path = segmentation_path.replace("\\", "/")
            response["segmentation_path"] = segmentation_web_path

        return jsonify(response), 201

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@doctor_bp.route("/validate/<prediction_id>", methods=["POST"])
@jwt_required()
def validate_prediction(prediction_id):
    try:
        doctor_id = get_jwt_identity()
        data = request.json or {}
        validated = data.get("validated", True)
        notes = data.get("notes", "")

        # Fetch prediction from DB
        prediction = doctor_bp.prediction_model.get_prediction_by_id(prediction_id)
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        if validated:
            # Only generate report if doctor validates
            patient = doctor_bp.user_model.find_user_by_id(prediction["patient_id"])
            doctor = doctor_bp.user_model.find_user_by_id(doctor_id)
            report_path = os.path.join("uploads/reports", f"{uuid.uuid4().hex}_report.pdf")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            generate_pdf_report(
                patient_name=patient.get("name", "Unknown"),
                patient_id=prediction["patient_id"],
                doctor_name=doctor.get("name", "Unknown"),
                predicted_class=prediction["predicted_class"],
                confidence=prediction["confidence"],
                mri_image_path = prediction["mri_image_path"],
                segment_image_path = prediction["segmentation_path"],
                notes=notes,
                save_path=report_path
            )

            # Update prediction with validated=True and PDF path
            success = doctor_bp.prediction_model.update_prediction(
                prediction_id,
                validated=True,
                notes=notes,
                report_pdf_path=report_path
            )
        else:
            # If not validated, just update validated field/notes
            success = doctor_bp.prediction_model.update_prediction(
                prediction_id,
                validated=False,
                notes=notes
            )

        if success:
            return jsonify({"message": "Prediction updated successfully"})
        else:
            return jsonify({"error": "Prediction not found"}), 404

    except Exception as e:
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500


@doctor_bp.route("/predictions", methods=["GET"])
@jwt_required()
def my_predictions():
    doctor_id = get_jwt_identity()
    preds = doctor_bp.prediction_model.get_predictions_by_doctor(doctor_id)
    result = []
    for p in preds:
        patient_name =doctor_bp.user_model.get_name_by_id(p['patient_id'])
        result.append({
            "_id": str(p["_id"]),
            "patient_name": patient_name,
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
    return jsonify(result)


@doctor_bp.route("/report/<prediction_id>", methods=["GET"])
@jwt_required()
def get_report(prediction_id):
    pred = doctor_bp.prediction_model.get_prediction_by_id(prediction_id)
    if not pred or not pred.get("report_pdf_path"):
        return jsonify({"error": "Report not found"}), 404
    return send_file(pred["report_pdf_path"], as_attachment=True)
