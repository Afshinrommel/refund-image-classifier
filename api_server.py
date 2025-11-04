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
    try:
        img = image.load_img(local_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        class_id = np.argmax(preds)
        confidence = float(np.max(preds))
        class_name = index_to_class[class_id] if index_to_class else str(class_id)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

    return jsonify({
        "predicted_class": class_name,
        "confidence": confidence,
        "uploaded_to": file_url
    })

# ===================================
# HTML UPLOAD PAGE
# ===================================
@app.route("/upload", methods=["GET"])
def upload_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Refund Image Classifier</title>
        <style>
            body { font-family: Arial; display:flex; flex-direction: column; align-items:center; justify-content:center; height:100vh; background:#f8f9fa;}
            .container { background:white; padding:30px; border-radius:12px; box-shadow:0 4px 8px rgba(0,0,0,0.1); text-align:center; width:350px;}
            input[type=file] { margin:15px 0; }
            button { background:#007bff; color:white; padding:10px 20px; border:none; border-radius:8px; cursor:pointer; }
            button:hover { background:#0056b3; }
            img { width:200px; margin-top:10px; border-radius:10px; display:none; }
            .result { margin-top:20px; font-weight:bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Refund Image Classifier</h2>
            <form id="uploadForm">
                <input type="file" name="image" id="imageInput" accept="image/*" required><br>
                <img id="preview" alt="Image Preview"><br>
                <button type="submit">Predict</button>
            </form>
            <div class="result" id="result"></div>
        </div>

        <script>
        const form = document.getElementById('uploadForm');
        const resultDiv = document.getElementById('result');
        const preview = document.getElementById('preview');
        const imageInput = document.getElementById('imageInput');

        imageInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => { preview.src = e.target.result; preview.style.display = 'block'; };
                reader.readAsDataURL(file);
                resultDiv.innerHTML = "";
            } else { preview.style.display = 'none'; }
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            resultDiv.innerHTML = "⏳ Predicting...";

            try {
                const response = await fetch("/predict", { method:"POST", body:formData });
                const data = await response.json();
                if (data.error) { resultDiv.innerHTML = "❌ "+data.error; }
                else {
                    resultDiv.innerHTML = `✅ <b>${data.predicted_class}</b><br>Confidence: ${(data.confidence*100).toFixed(2)}%<br>` +
                                          (data.uploaded_to ? `<a href="${data.uploaded_to}" target="_blank">📂 View in Google Drive</a>` : "");
                }
            } catch (err) {
                resultDiv.innerHTML = "⚠️ Error: "+err.message;
            }
        });
        </script>
    </body>
    </html>
    """

# ===================================
# MAIN
# ===================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
