import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# 🔧 CONFIGURATION
# ===============================
MODEL_DIR = r"C:/Un/Semster 4/Book/model to production/refund-image-classifier/model_saved"
CLASS_MAP_FILENAME = r"C:/Un/Semster 4/Book/model to production/refund-image-classifier/class_indices.json"

# ===============================
# 🧠 LOAD MODEL
# ===============================
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model folder '{MODEL_DIR}' not found.")

print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_DIR)
print("✅ Model loaded successfully!")

# Load class indices if available
if os.path.exists(CLASS_MAP_FILENAME):
    with open(CLASS_MAP_FILENAME, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
else:
    print("⚠️ Warning: class_indices.json not found. Using numeric class IDs.")
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
    try:
        # Read the uploaded file bytes
        file_bytes = file.read()
        img_bytes = io.BytesIO(file_bytes)  # wrap bytes in BytesIO for load_img

        # Load and preprocess image
        img = load_img(img_bytes, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict
        preds = model.predict(x)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        class_name = index_to_class.get(class_id, str(class_id))

        return jsonify({
            "predicted_class": class_name,
            "confidence": confidence
        })

    except Exception as e:
        import traceback
        print(f"⚠️ Error processing file {file.filename}:\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Failed to process image {file.filename}",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port)
