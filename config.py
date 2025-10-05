import os

MONGO_URI = "mongodb://localhost:27017/brain_tumor_db"
SECRET_KEY = "Code never lies, comments sometimes do"
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
REPORT_FOLDER = os.path.join(os.getcwd(), "reports")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
EFF_MODEL_PATH = os.path.join(os.getcwd(), "best_EfficientNetB0.keras")  # Your trained model
UNET_MODEL_PATH = os.path.join(os.getcwd(), "3D_MRI_Brain_tumor_segmentation.h5")