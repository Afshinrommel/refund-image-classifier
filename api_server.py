import os
import json
from flask import Flask, request, jsonify, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown
from werkzeug.utils import secure_filename
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ===================================
# CONFIGURATION
# ===================================
MODEL_URL = "https://drive.google.com/uc?id=10QVwxFBdz6nhqQoe-LnQSQi87e2kPRLv"
MODEL_FILENAME = "model.h5"
CLASS_MAP_FILENAME = "class_indices.json"
DRIVE_FOLDER_ID = "1i1jDWnm2mi4MCWIHLduIyVQkGcJv_Uvg"  # فولدر اشتراک

# ===================================
# CREATE service_account.json FROM ENV
# ===================================
if os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"):
    with open("service_account.json", "w") as f:
        f.write(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
    print("✅ service_account.json created from environment variable")
else:
    raise Exception("⚠️ GOOGLE_SERVICE_ACCOUNT_JSON environment variable missing. Drive upload will fail!")

# ===================================
# DOWNLOAD MODEL
# ===================================
def ensure_model():
    if not os.path.exists(MODEL_FILENAME):
        print("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)
    print("✅ Model ready:", MODEL_FILENAME)

ensure_model()
model = tf.keras.models.load_model(MODEL_FILENAME)
print("✅ Model loaded successfully")

# ===================================
# LOAD CLASS MAP
# ===================================
if os.path.exists(CLASS_MAP_FILENAME):
    with open(CLASS_MAP_FILENAME) as f:
        index_to_class = {v: k for k, v in json.load(f).items()}
else:
    index_to_class = None

# ===================================
# GOOGLE DRIVE SETUP
# ===================================
def setup_drive():
    gauth = GoogleAuth()
    try:
        gauth.ServiceAuth(filename="service_account.json")  # authenticate با service account
    except Exception as e:
        print(f"⚠️ Failed to authenticate Drive: {e}")
        return None
    drive = GoogleDrive(gauth)
    return drive

drive = setup_drive()
if drive:
    print("✅ Google Drive connected")
else:
    print("⚠️ Google Drive not connected. Uploads will fail.")

# ===================================
# FLASK APP
# ===================================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return redirect(url_for('upload_page'))

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "message": "Refund Image Classifier API is running!",
        "usage": "POST /predict with a file field named 'image'",
        "ui": "/upload"
    })

# ===================================
# PREDICT + UPLOAD
# ===================================
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    if img_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs("uploads", exist_ok=True)
    filename = secure_filename(img_file.filename)
    local_path = os.path.join("uploads", filename)
    img_file.save(local_path)

    # Upload to Drive
    file_url = None
    if drive:
        try:
            file_drive = drive.CreateFile({
                'title': filename,
                'parents': [{'id': DRIVE_FOLDER_ID}]
            })
            file_drive.SetContentFile(local_path)
            file_drive.Upload()
            file_drive.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})
            file_url = f"https://drive.google.com/uc?id={file_drive['id']}"
            print(f"📤 Uploaded to Drive: {file_url}")
        except Exception as e:
            return jsonify({"error": f"Failed to upload to Drive: {str(e)}"}), 500

    # Predict
