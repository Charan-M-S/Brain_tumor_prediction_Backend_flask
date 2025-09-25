from fpdf import FPDF
from datetime import datetime
import os
def generate_pdf_report(patient_name, patient_id, doctor_name, predicted_class, confidence, notes, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brain Tumor Prediction Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 10, f"Doctor Name: {doctor_name}", ln=True)
    pdf.cell(0, 10, f"Predicted Class: {predicted_class}", ln=True)
    pdf.cell(0, 10, f"Model Confidence: {confidence*100:.2f}%", ln=True)
    pdf.multi_cell(0, 10, f"Validation Notes: {notes}")
    pdf.cell(0, 10, f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
    pdf.output(save_path)
