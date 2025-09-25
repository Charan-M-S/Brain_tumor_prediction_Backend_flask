from datetime import datetime
from bson import ObjectId

class Prediction:
    def __init__(self, mongo):
        self.collection = mongo.db.predictions

    def create_prediction(self, patient_id, doctor_id, image_path, predicted_class, confidence, validated=False, notes="", report_pdf_path=None):
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
            "created_at": datetime.utcnow()
        }
        inserted = self.collection.insert_one(prediction_data)
        return str(inserted.inserted_id)

    def update_prediction(self, prediction_id, validated=True, notes=""):
        result = self.collection.update_one(
            {"_id": ObjectId(prediction_id)},
            {"$set": {"validated": validated, "notes": notes, "status": "completed"}}
        )
        return result.modified_count > 0

    def get_predictions_by_doctor(self, doctor_id):
        return list(self.collection.find({"doctor_id": ObjectId(doctor_id)}))

    def get_predictions_by_patient(self, patient_id):
        return list(self.collection.find({"patient_id": ObjectId(patient_id)}))

    def get_prediction_by_id(self, prediction_id):
        return self.collection.find_one({"_id": ObjectId(prediction_id)})
