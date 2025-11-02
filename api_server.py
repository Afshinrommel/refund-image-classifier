import os
import json
import gdown
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

# ===============================
# 🔧 CONFIGURATION
# ===============================
MODEL_URL = "https://drive.google.com/uc?id=10QVwxFBdz6nhqQoe-LnQSQi87e2kPRLv"
MODEL_FILENAME = "model.h5"
CLASS_MAP_FILENAME = "class_indices.json"

# ===============================
# 📥 DOWNLOAD MODEL IF NEEDED
# ===============================
def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"⬇️ Downloading model from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        print("✅ Model downloaded!")

download_file(MODEL_URL, MODEL_FILENAME)

# ===============================
# 🧠 LOAD MODEL AND CLASSES
# ===============================
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_FILENAME)
print("✅ Model loaded successfully!")

if os.path.exists(CLASS_MAP_FILENAME):
    with open(CLASS_MAP_FILENAME, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
else:
    index_to_class = {}

# ===============================
# 🚀 FLASK APP
# ===============================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Refund Classifier API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    class_name = index_to_class.get(class_id, str(class_id))

    return jsonify({
        "predicted_class": class_name,
        "confidence": confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
