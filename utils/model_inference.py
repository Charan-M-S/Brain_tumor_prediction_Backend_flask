from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
from keras.models import load_model
import numpy as np
from config import MODEL_PATH
from fpdf import FPDF
from datetime import datetime
import os

# Class names for prediction
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Global model variable
model = None

def load_brain_model():
    """Load the .keras model only once."""
    global model
    if model is None:
        print("⏳ Loading model...")
        model = load_model(MODEL_PATH)  # Ensure MODEL_PATH points to your .keras file
        print("✅ Model loaded successfully!")

def predict_mri(image_path, target_size=(224, 224)):
    """
    Predicts the class of the MRI image.
    Returns: predicted class and confidence
    """
    if model is None:
        load_brain_model()
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds, axis=-1)[0]
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(preds[0][predicted_index])

    return predicted_class, confidence

def generate_pdf_report(patient_name, patient_id, doctor_name, predicted_class, confidence, notes, save_path):
    """
    Generates a PDF report for the prediction
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brain Tumor Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Patient & doctor info
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 10, f"Doctor Name: {doctor_name}", ln=True)

    # Prediction result
    pdf.cell(0, 10, f"Predicted Class: {predicted_class}", ln=True)
    pdf.cell(0, 10, f"Model Confidence: {confidence*100:.2f}%", ln=True)

    # Notes
    pdf.multi_cell(0, 10, f"Validation Notes: {notes}")

    # Report timestamp
    pdf.cell(0, 10, f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)

    # Save PDF
    pdf.output(save_path)
    print(f"✅ PDF report saved at {save_path}")
