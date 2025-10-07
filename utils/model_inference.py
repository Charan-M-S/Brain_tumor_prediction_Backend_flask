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
CLASS_NAMES = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
# Global models
eff_model = None
unet_model = None

# --- Load Models --- #
def load_models():
    global eff_model, unet_model
    if eff_model is None:
        print("⏳ Loading EfficientNet model...")
        eff_model = load_model(EFF_MODEL_PATH,compile=False)
        print("✅ EfficientNet loaded.")

    if unet_model is None:
        print("⏳ Loading U-Net model...")
        # unet_model = load_model(
        #     UNET_MODEL_PATH,
        #     custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou_coef': iou_coef}
        # )
        # print("✅ U-Net loaded.")
        unet_model = load_model(UNET_MODEL_PATH,compile=False)
        print("✅ U-Net loaded.")

# --- Predict MRI Class --- #
def predict_mri(image_path, target_size=(224, 224)):
    if eff_model is None:
        load_models()

    # img = image.load_img(image_path, target_size=target_size)
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img_array = np.expand_dims(img, axis=0)

    preds = eff_model.predict(img_array)
    predicted_index = np.argmax(preds, axis=-1)[0]
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(preds[0][predicted_index])

    return predicted_class, confidence

def preprocess_mri(image_path, img_size=128):
    # Load grayscale MRI
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to model input
    img = cv2.resize(img, (img_size, img_size))
    # Normalize
    img = img / 255.0
    # Expand dims to create shape (128,128,1)
    img = np.expand_dims(img, axis=-1)
    # Duplicate channel to match model input (2 channels)
    img = np.concatenate([img, img], axis=-1)  # shape -> (128,128,2)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # shape -> (1,128,128,2)
    return img

def segment_mri1(image_path, save_dir="uploads/segmented"):
    if unet_model is None:
        load_models()

    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------
    # 1. Load and preprocess image
    # -------------------------------
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_norm = img_gray / 255.0
    img_input = np.expand_dims(img_norm, axis=-1)
    img_input = np.concatenate([img_input, img_input], axis=-1)  # shape -> (128,128,2)
    img_input = np.expand_dims(img_input, axis=0)  # Add batch

    # -------------------------------
    # 2. Predict mask
    # -------------------------------
    pred_mask = unet_model.predict(img_input)[0]  # shape -> (128,128,num_classes)
    mask_class = np.argmax(pred_mask, axis=-1)

    # -------------------------------
    # 3. Create colored overlay
    # -------------------------------
    colors = [
        [0, 0, 0],       # background
        [255, 0, 0],     # class 1
        [0, 255, 0],     # class 2
        [0, 0, 255],     # class 3
    ]
    colored_mask = np.zeros((mask_class.shape[0], mask_class.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored_mask[mask_class == i] = color

    overlay = cv2.addWeighted(img_resized, 0.7, colored_mask, 0.3, 0)

    # -------------------------------
    # 4. Save overlay image
    # -------------------------------
    filename = f"{uuid.uuid4().hex}_overlay.png"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, overlay)

    return save_path
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
        segmentation_path = segment_mri1(image_path)

    return predicted_class, confidence, segmentation_path
