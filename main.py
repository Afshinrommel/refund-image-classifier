import os
import json
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown
import re

# ===============================
# CONFIGURATION SECTION
# ===============================
# Path to  saved_model.pb
MODEL_DIR = "model_saved"

CLASS_MAP_URL = "https://drive.google.com/uc?id=1RL6oiB8UOVkOmCsjqO5qbJsS0g7ZFMqf"
CLASS_MAP_FILENAME = "class_indices.json"

TEST_FOLDER_URL = "https://drive.google.com/drive/folders/1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc?usp=drive_link"
TEST_DIR = "test_images"

PREDICTION_DIR = "predictions"
os.makedirs(PREDICTION_DIR, exist_ok=True)

# ===============================
# DOWNLOAD TEST IMAGES
# ===============================
def download_test_images():
    print("⬇️ Downloading latest test images from Google Drive...")
    gdown.download_folder(url=TEST_FOLDER_URL, output=TEST_DIR, quiet=False, use_cookies=False)
    print("✅ Test images downloaded!")

# ===============================
# LOAD MODEL & CLASSES
# ===============================
def load_model_or_exit(path):
    """
    Loads the local SavedModel directory. 
    Ensures the model used is exactly the one from the local environment.
    """
    if not os.path.exists(path):
        print(f"⚠️ Error: Model directory '{path}' not found.")
        print("Ensure 'model_saved' folder is present in the repository.")
        exit(1)
        
    print(f"📦 Loading local model from: {path}")
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
# PREDICTION FUNCTION
# ===============================
def predict_image(model, img_path, index_to_class):
    # Standard MobileNetV2 input size: 224x224
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
# RUN BATCH PREDICTION WITH VALIDATION
# ===============================
def run_batch():
    print(f"🕐 Starting batch run: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    download_test_images()

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"No directory named '{TEST_DIR}' found.")

    # Load from the local SavedModel directory
    model = load_model_or_exit(MODEL_DIR)
    index_to_class = load_class_map()

    csv_file = os.path.join(PREDICTION_DIR, "predictions.csv")
    
    # Enhanced column structure for Professor's requirements
    columns = [
        "filename", "true_label", "predicted_class_name", 
        "is_match", "customer_name", "customer_id", 
        "confidence", "timestamp"
    ]
    
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
    else:
        df_existing = pd.DataFrame(columns=columns)

    new_records = []
    for fname in os.listdir(TEST_DIR):
        fpath = os.path.join(TEST_DIR, fname)
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # Advanced Regex to handle complex filenames (e.g., watch_1_Amir_11111.png)
        # Captures: [Object Label], [Customer Name], [Customer ID]
        match = re.match(r"([A-Za-z]+)(?:_\d+)?_([A-Za-z]+)_(\d+)", fname)
        
        if match:
            true_label = match.group(1).lower()
            cust_name = match.group(2)
            cust_id = match.group(3)
        else:
            true_label = "unknown"
            cust_name = fname
            cust_id = "0"

        class_id, class_name, confidence = predict_image(model, fpath, index_to_class)

        # Accuracy Validation: Comparing filename label with AI prediction
        is_match = true_label in class_name.lower()

        # Check if record exists with same filename and prediction to avoid duplicates
        existing = df_existing[
            (df_existing["filename"] == fname) &
            (df_existing["predicted_class_name"] == class_name) &
            (np.isclose(df_existing["confidence"], confidence, atol=1e-4))
        ]

        if existing.empty:
            new_records.append({
                "filename": fname,
                "true_label": true_label,
                "predicted_class_name": class_name,
                "is_match": is_match,
                "customer_name": cust_name,
                "customer_id": cust_id,
                "confidence": round(confidence, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            })
            print(f"🖼️ Processed: {fname} -> AI: {class_name} | Match: {is_match}")
        else:
            print(f"ℹ️ Skipping existing entry: {fname}")

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Final column alignment
        df_combined = df_combined[columns]
        df_combined.to_csv(csv_file, index=False)
        print(f"✅ CSV updated successfully: {csv_file}")
    else:
        print("⚠️ No new unique images detected for processing.")

# ===============================
# MAIN ENTRY POINT
# ===============================
if __name__ == "__main__":
    print(">>> Automated Batch Prediction System (v5.5 - Research Mode) <<<")
    run_batch()