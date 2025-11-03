import os
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
        "usage": "POST /predict with a file field named 'image'"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    img = image.load_img(img_file, target_size=(224, 224))
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
