from flask import Blueprint, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity

patient_bp = Blueprint("patient", __name__)

# These are set from app.py when you call inject_models()
user_model = None
prediction_model = None

@patient_bp.route("/predictions", methods=["GET"])
@jwt_required()
def my_predictions():
    """
    Return all predictions for the logged-in patient
    """
    # identity is the patient's id (as stored in JWT)
    patient_id = get_jwt_identity()

    # get all predictions for this patient
    preds = patient_bp.prediction_model.get_predictions_by_patient(patient_id)

    result = []
    for p in preds:
        p["_id"] = str(p["_id"])
        if "patient_id" in p:
            p["patient_id"] = str(p["patient_id"])
        if "doctor_id" in p:
            p["doctor_id"] = str(p["doctor_id"])
        result.append(p)

    return jsonify(result), 200


@patient_bp.route("/report/<prediction_id>", methods=["GET"])
def download_report(prediction_id):
    pred = patient_bp.prediction_model.get_prediction_by_id(prediction_id)
    if not pred or not pred.get("report_pdf_path"):
        return jsonify({"error": "Report not found"}), 404
    return send_file(pred["report_pdf_path"], as_attachment=True)
