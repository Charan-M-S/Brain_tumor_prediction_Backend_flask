# model_inference.py

import os
import uuid
from datetime import datetime
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
from keras import backend as K
from keras.utils import register_keras_serializable
from fpdf import FPDF
import cv2

# Paths to your models
from config import EFF_MODEL_PATH, UNET_MODEL_PATH

# Class names for EfficientNet
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Global models
eff_model = None
unet_model = None

# --- Custom Metrics / Losses --- #
@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

@register_keras_serializable()
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum_val = K.sum(y_true + y_pred)
    return (intersection + smooth) / (sum_val - intersection + smooth)

# --- Load Models --- #
def load_models():
    global eff_model, unet_model
    if eff_model is None:
        print("⏳ Loading EfficientNet model...")
        eff_model = load_model(EFF_MODEL_PATH)
        print("✅ EfficientNet loaded.")

    if unet_model is None:
        print("⏳ Loading U-Net model...")
        unet_model = load_model(
            UNET_MODEL_PATH,
            custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou_coef': iou_coef}
        )
        print("✅ U-Net loaded.")

# --- Predict MRI Class --- #
def predict_mri(image_path, target_size=(224, 224)):
    if eff_model is None:
        load_models()

    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = eff_model.predict(img_array)
    predicted_index = np.argmax(preds, axis=-1)[0]
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(preds[0][predicted_index])

    return predicted_class, confidence

# --- Run U-Net Segmentation --- #
def segment_mri(image_path, save_dir="uploads/segmented"):
    if unet_model is None:
        load_models()

    os.makedirs(save_dir, exist_ok=True)

    # Load image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Predict mask
    pred_mask = unet_model.predict(img_input)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Binary mask

    # Save mask image
    mask_filename = f"{uuid.uuid4().hex}_mask.png"
    mask_path = os.path.join(save_dir, mask_filename)
    cv2.imwrite(mask_path, pred_mask)

    return mask_path

# --- Generate PDF Report --- #
from fpdf import FPDF
from datetime import datetime
import os

def generate_pdf_report(
    patient_name,
    patient_id,
    doctor_name,
    predicted_class,
    confidence,
    notes,
    save_path,
    mri_image_path=None,  # ✅ added
    segment_image_path=None
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brain Tumor Prediction Report", ln=True, align="C")
    pdf.ln(10)
    # ✅ MRI Image (optional)
    if mri_image_path and os.path.exists(mri_image_path):
        pdf.ln(10)  # add some space
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "MRI Scan:", ln=True)
        pdf.ln(5)
        # width = 100 mm automatically keeps aspect ratio if height not given
        pdf.image(mri_image_path, x=None, y=None, w=100)


    # Details
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 10, f"Doctor Name: {doctor_name}", ln=True)
    pdf.cell(0, 10, f"Predicted Class: {predicted_class}", ln=True)
    pdf.cell(0, 10, f"Model Confidence: {confidence*100:.2f}%", ln=True)
    pdf.multi_cell(0, 10, f"Validation Notes: {notes}")
    pdf.cell(0, 10, f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
    
    if segment_image_path and os.path.exists(segment_image_path):
        pdf.ln(50)  # add some space
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Segment Mask:", ln=True)
        pdf.ln(5)
        # width = 100 mm automatically keeps aspect ratio if height not given
        pdf.image(segment_image_path, x=None, y=None, w=100)

    pdf.output(save_path)

# --- Full Pipeline --- #
def predict_and_segment(image_path):
    """
    Returns:
        predicted_class, confidence, segmentation_path (if applicable)
    """
    predicted_class, confidence = predict_mri(image_path)
    segmentation_path = None

    if predicted_class != "no_tumor":
        segmentation_path = segment_mri(image_path)

    return predicted_class, confidence, segmentation_path
