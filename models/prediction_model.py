from datetime import datetime
from bson import ObjectId
from models.user_model import User

class Prediction:
    def __init__(self, mongo):
        self.collection = mongo.db.predictions
        self.user_model = User(mongo)

    def create_prediction(self, patient_id, doctor_id, image_path, predicted_class, confidence, validated=False, notes="", report_pdf_path=None,segmentation_path=None):
        prediction_data = {
            "patient_id": ObjectId(patient_id),
            "doctor_id": ObjectId(doctor_id),
            "mri_image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "validated": validated,
            "notes": notes,
            "report_pdf_path": report_pdf_path,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "segmentation_path":segmentation_path
        }
        inserted = self.collection.insert_one(prediction_data)
        return str(inserted.inserted_id)

    def update_prediction(self, prediction_id, validated=True, notes="",report_pdf_path=None):
        result = self.collection.update_one(
            {"_id": ObjectId(prediction_id)},
            {"$set": {"validated": validated, "notes": notes, "status": "completed","report_pdf_path":report_pdf_path}}
        )
        return result.modified_count > 0

    def get_predictions_by_doctor(self, doctor_id):
        return list(self.collection.find({"doctor_id": ObjectId(doctor_id)}))

    def get_predictions_by_patient(self, patient_id):
        predictions = list(self.collection.find({"patient_id": ObjectId(patient_id)}))
        enriched = []

        for pred in predictions:
            doctor_name = self.user_model.get_name_by_id(pred["doctor_id"])
            patient_name = self.user_model.get_name_by_id(pred["patient_id"])

            enriched.append({
                "_id": str(pred["_id"]),
                "patient_name": patient_name,
                "doctor_name": doctor_name,
                "mri_image_path": pred["mri_image_path"],
                "predicted_class": pred["predicted_class"],
                "confidence": pred["confidence"],
                "validated": pred["validated"],
                "notes": pred.get("notes", ""),
                "report_pdf_path": pred.get("report_pdf_path"),
                "status": pred.get("status"),
                "created_at": pred.get("created_at"),
                "segmentation_path":pred.get("segmentation_path")
            })
        return enriched
    
    def get_prediction_by_id(self, prediction_id):
        return self.collection.find_one({"_id": ObjectId(prediction_id)})
