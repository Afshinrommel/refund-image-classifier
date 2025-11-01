from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
from PIL import Image
import base64
import json
import os
import gdown

app = Flask(__name__)

# ===============================
# 🔧 CONFIG
# ===============================
MODEL_URL = "https://drive.google.com/uc?id=10QVwxFBdz6nhqQoe-LnQSQi87e2kPRLv"  # Google Drive link
MODEL_FILENAME = "model.h5"
CLASS_MAP_FILENAME = "class_indices.json"

# ===============================
# 📥 Download Model if Missing
# ===============================
def download_if_needed():
    if not os.path.exists(MODEL_FILENAME):
        print(f"⬇️ Downloading model from {MODEL_URL} ...")
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)
        print("✅ Model downloaded successfully!")

download_if_needed()

# ===============================
# 🧠 Load Model and Class Map
# ===============================
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_FILENAME)
print("✅ Model loaded!")

if os.path.exists(CLASS_MAP_FILENAME):
    with open(CLASS_MAP_FILENAME, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    print("✅ Loaded class mapping file.")
else:
    index_to_class = None
    print("⚠️ class_indices.json not found; returning class IDs only.")


# ===============================
# 🧩 Helper Function
# ===============================
def prepare_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# ===============================
# 🚀 API Endpoints
# ===============================
@app.route("/")
def home():
    return jsonify({
        "message": "Refund Image Classifier API is running ✅",
        "usage": "POST /predict with base64 images to get predictions"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    if "images" not in data:
        return jsonify({"error": "Missing key 'images'"}), 400

    results = []
    for img_str in data["images"]:
        try:
            img_bytes = base64.b64decode(img_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            x = prepare_image(img)
            preds = model.predict(x)
            class_id = int(np.argmax(preds))
            confidence = float(np.max(preds))
            probs = preds[0].tolist()

            result = {
                "class_id": class_id,
                "class_name": index_to_class[class_id] if index_to_class else str(class_id),
                "confidence": round(confidence, 4),
                "probabilities": {index_to_class[i]: float(p) for i, p in enumerate(probs)} if index_to_class else probs
            }

            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
