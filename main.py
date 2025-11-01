import os
import json
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown

# ===============================
# 🧠 CONFIGURATION SECTION
# ===============================
MODEL_URL = "https://drive.google.com/uc?id=10QVwxFBdz6nhqQoe-LnQSQi87e2kPRLv"  # model.h5 in Drive
MODEL_FILENAME = "model.h5"
CLASS_MAP_FILENAME = "class_indices.json"

# ✅ Shared Google Drive folder (with test images)
# -> share the folder as "Anyone with the link" and replace the ID below
TEST_FOLDER_URL = "https://drive.google.com/drive/folders/1owlRknDThdlDVwOzpFmc_tqS5qRTXS5J"
TEST_DIR = "test_images"

PREDICTION_DIR = "predictions"
os.makedirs(PREDICTION_DIR, exist_ok=True)

# ===============================
# 🔽 DOWNLOAD TEST IMAGES
# ===============================
def download_test_images():
    """Download test images from Google Drive folder."""
    print("⬇️ Downloading latest test images from Google Drive...")
    gdown.download_folder(url=TEST_FOLDER_URL, output=TEST_DIR, quiet=False, use_cookies=False)
    print("✅ Test images downloaded!")

# ===============================
# 🧩 LOAD MODEL & CLASSES
# ===============================
def download_file(url, output_path):
    """Download model if missing."""
    import gdown
    print(f"⬇️ Downloading model from: {url}")
    gdown.download(url, output_path, quiet=False)
    print(f"✅ Model downloaded: {output_path}")

def load_model_or_exit(path):
    if not os.path.exists(path):
        print("⚠️ Model not found locally. Downloading...")
        download_file(MODEL_URL, path)
    print("📦 Loading model...")
    model = tf.keras.models.load_model(path)
    print("✅ Model loaded successfully!")
    return model

def load_class_map():
    if not os.path.exists(CLASS_MAP_FILENAME):
        print("⚠️ class_indices.json not found — classes will show as numbers.")
        return None
    with open(CLASS_MAP_FILENAME, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

# ===============================
# 🔍 PREDICTION FUNCTION
# ===============================
def predict_image(model, img_path, index_to_class):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    class_name = index_to_class[class_id] if index_to_class else str(class_id)

    return class_id, class_name, confidence

# ===============================
# 🚀 RUN BATCH PREDICTION
# ===============================
def run_batch():
    print(f"🕐 Starting batch run: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    download_test_images()  # ✅ always refresh test images

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"No directory named '{TEST_DIR}' found.")

    model = load_model_or_exit(MODEL_FILENAME)
    index_to_class = load_class_map()

    results = []
    for fname in os.listdir(TEST_DIR):
        fpath = os.path.join(TEST_DIR, fname)
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        class_id, class_name, confidence = predict_image(model, fpath, index_to_class)
        results.append({
            "filename": fname,
            "predicted_class_id": class_id,
            "predicted_class_name": class_name,
            "confidence": round(confidence, 4)
        })
        print(f"🖼️ {fname} → {class_name} ({confidence:.2f})")

    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        output_file = os.path.join(PREDICTION_DIR, f"predictions_{timestamp}.csv")
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"✅ Predictions saved to {output_file}")
    else:
        print("⚠️ No images found to process.")

# ===============================
# 🏁 MAIN ENTRY
# ===============================
if __name__ == "__main__":
    print(">>> Running Automated Batch Prediction (v3.0) <<<")
    run_batch()
