from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId

class User:
    def __init__(self, mongo):
        self.collection = mongo.db.users

    def find_user(self, email):
        return self.collection.find_one({"email": email})

    def create_user(self, name, email, password, role, assigned_doctor_id=None):
        if self.collection.find_one({"email": email}):
            return None
        user = {
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "role": role
        }
        if role == "patient" and assigned_doctor_id:
            user["assigned_doctor_id"] = ObjectId(assigned_doctor_id)
        inserted = self.collection.insert_one(user)
        return str(inserted.inserted_id)

    def verify_password(self, email, password):
        user = self.collection.find_one({"email": email})
        if not user:
            return None
        if check_password_hash(user["password_hash"], password):
            return user
        return None

    def reset_password(self, user_id, new_password):
        result = self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"password_hash": generate_password_hash(new_password)}}
        )
        return result.modified_count > 0

    def find_user_by_id(self, user_id):
        return self.collection.find_one({"_id": ObjectId(user_id)})

    def get_patients_of_doctor(self, doctor_id):
        return list(self.collection.find({"role": "patient", "assigned_doctor_id": ObjectId(doctor_id)}, {"password_hash": 0}))
