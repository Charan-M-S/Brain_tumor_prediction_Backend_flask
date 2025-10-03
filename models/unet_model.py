import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime

# load .keras model
UNET_MODEL = tf.keras.models.load_model("models/unet_model.keras")

def run_unet_segmentation(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    predicted_mask = UNET_MODEL.predict(img_input)[0]
    mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    mask_full = cv2.resize(mask, (img.shape[1], img.shape[0]))

    overlay = img.copy()
    overlay[mask_full == 255] = [0, 0, 255]  # red region
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    out_dir = os.path.join("uploads", "segmentations")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_segmented.jpg"
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, blended)

    return out_path
