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
MODEL_URL = "https://drive.google.com/uc?id=154facxUyy9kohASgOmUOBixIKswRhNPI"   # model.h5 in Drive
MODEL_FILENAME = "model.h5"

CLASS_MAP_URL = "https://drive.google.com/uc?id=1RL6oiB8UOVkOmCsjqO5qbJsS0g7ZFMqf" # model_CLASS.JSON in Drive
CLASS_MAP_FILENAME = "class_indices.json"

TEST_FOLDER_URL = "https://drive.google.com/drive/folders/1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc?usp=drive_link"
TEST_DIR = "test_images"

PREDICTION_DIR = "predictions"
os.makedirs(PREDICTION_DIR, exist_ok=True)

# ===============================
# 🔽 DOWNLOAD TEST IMAGES
# ===============================
def download_test_images():
    '''Download test images from Google Drive folder.'''
    print("⬇️ Downloading latest test images from Google Drive...")
    gdown.download_folder(url=TEST_FOLDER_URL, output=TEST_DIR, quiet=False, use_cookies=False)
    print("✅ Test images downloaded!")

# ===============================
# 🧩 LOAD MODEL & CLASSES
# ===============================
def download_file(url, output_path):
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
        print("⚠️ Class map not found. Downloading...")
        gdown.download(CLASS_MAP_URL, CLASS_MAP_FILENAME, quiet=False)
        print("✅ Class map downloaded:", CLASS_MAP_FILENAME)

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
# 🚀 RUN BATCH PREDICTION (UPDATED)
# ===============================
def run_batch():
    print(f"🕐 Starting batch run: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    download_test_images()

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"No directory named '{TEST_DIR}' found.")

    model = load_model_or_exit(MODEL_FILENAME)
    index_to_class = load_class_map()

    # Load existing CSV if exists
    csv_file = os.path.join(PREDICTION_DIR, "predictions.csv")
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
    else:
        df_existing = pd.DataFrame(columns=["filename","predicted_class_name","confidence","timestamp"])

    new_records = []
    for fname in os.listdir(TEST_DIR):
        fpath = os.path.join(TEST_DIR, fname)
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        class_id, class_name, confidence = predict_image(model, fpath, index_to_class)

        # Check if this record already exists with same filename, class, and confidence
        existing = df_existing[
            (df_existing["filename"] == fname) &
            (df_existing["predicted_class_name"] == class_name) &
            (np.isclose(df_existing["confidence"], confidence, atol=1e-4))
        ]
        if existing.empty:
            # New or updated record → add timestamp
            new_records.append({
                "filename": fname,
                "predicted_class_name": class_name,
                "confidence": round(confidence, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            })
            print(f"🖼️ New/Updated: {fname} → {class_name} ({confidence:.2f})")
        else:
            print(f"ℹ️ Skipping existing: {fname} → {class_name} ({confidence:.2f})")

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
        print(f"✅ CSV updated: {csv_file}")
    else:
        print("⚠️ No new or updated images to process.")

# ===============================
# 🏁 MAIN ENTRY
# ===============================
if __name__ == "__main__":
    print(">>> Running Automated Batch Prediction (v4.0) <<<")
    run_batch()
