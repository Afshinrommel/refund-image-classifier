import os
import io
import json
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown

# ===================================
# CONFIGURATION
# ===================================
MODEL_URL = "https://drive.google.com/uc?id=10QVwxFBdz6nhqQoe-LnQSQi87e2kPRLv"  # Replace with your model URL
MODEL_FILENAME = "model.h5"
CLASS_MAP_FILENAME = "class_indices.json"

# ===================================
# DOWNLOAD MODEL (if missing)
# ===================================
def ensure_model():
    if not os.path.exists(MODEL_FILENAME):
        print("⬇️ Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)
    print("✅ Model ready:", MODEL_FILENAME)

ensure_model()
model = tf.keras.models.load_model(MODEL_FILENAME)
print("✅ Model loaded successfully")

# Load class map
if os.path.exists(CLASS_MAP_FILENAME):
    with open(CLASS_MAP_FILENAME) as f:
        index_to_class = {v: k for k, v in json.load(f).items()}
else:
    index_to_class = None

# ===================================
# FLASK APP
# ===================================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Refund Image Classifier API is running!",
        "usage": "POST /predict with a file field named 'image'",
        "ui": "/upload"
    })

# ===================================
# PREDICTION ENDPOINT (POST)
# ===================================
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    if img_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = image.load_img(io.BytesIO(img_file.read()), target_size=(224, 224))
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {str(e)}"}), 400

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))
    class_name = index_to_class[class_id] if index_to_class else str(class_id)

    return jsonify({
        "predicted_class": class_name,
        "confidence": confidence
    })

# ===================================
# HTML UPLOAD PAGE (GET)
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
            body {
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                background-color: #f8f9fa;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                text-align: center;
                width: 350px;
            }
            input[type=file] {
                margin: 15px 0;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            img {
                width: 200px;
                margin-top: 10px;
                border-radius: 10px;
                display: none;
            }
            .result {
                margin-top: 20px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Refund Image Classifier</h2>
            <form id="uploadForm">
                <input type="file" name="image" id="imageInput" accept="image/*" required>
                <br>
                <img id="preview" alt="Image Preview">
                <br>
                <button type="submit">Predict</button>
            </form>
            <div class="result" id="result"></div>
        </div>

        <script>
        const form = document.getElementById('uploadForm');
        const resultDiv = document.getElementById('result');
        const preview = document.getElementById('preview');
        const imageInput = document.getElementById('imageInput');

        // Show image preview
        imageInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            } else {
                preview.style.display = 'none';
            }
        });

        // Handle form submit
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            resultDiv.innerHTML = "⏳ Predicting...";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();
                if (data.error) {
                    resultDiv.innerHTML = "❌ " + data.error;
                } else {
                    resultDiv.innerHTML = `✅ <b>${data.predicted_class}</b><br>Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                }
            } catch (err) {
                resultDiv.innerHTML = "⚠️ Error: " + err.message;
            }
        });
        </script>
    </body>
    </html>
    """

# ===================================
# MAIN ENTRY POINT
# ===================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
