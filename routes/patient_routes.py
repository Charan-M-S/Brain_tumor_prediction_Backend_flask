from flask import Blueprint, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity

patient_bp = Blueprint("patient", __name__)
prediction_model = None  # injected
user_model = None        # injected

@patient_bp.route("/predictions", methods=["GET"])
@jwt_required()
def my_predictions():
    patient_id = get_jwt_identity()
    preds = prediction_model.get_predictions_by_patient(patient_id)
    result = []
    for p in preds:
        p["_id"] = str(p["_id"])
        p["patient_id"] = str(p["patient_id"])
        p["doctor_id"] = str(p["doctor_id"])
        result.append(p)
    return jsonify(result)

@patient_bp.route("/report/<prediction_id>", methods=["GET"])
@jwt_required()
def download_report(prediction_id):
    pred = prediction_model.get_prediction_by_id(prediction_id)
    if not pred or not pred.get("report_pdf_path"):
        return jsonify({"error": "Report not found"}), 404
    return send_file(pred["report_pdf_path"], as_attachment=True)
