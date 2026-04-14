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
# 🧠 CONFIGURATION SECTION
# ===============================
# Path to the local model folder (saved_model.pb)
MODEL_DIR = "model_saved"

CLASS_MAP_URL = "https://drive.google.com/uc?id=1RL6oiB8UOVkOmCsjqO5qbJsS0g7ZFMqf"
CLASS_MAP_FILENAME = "class_indices.json"

TEST_FOLDER_URL = "https://drive.google.com/drive/folders/1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc?usp=drive_link"
TEST_DIR = "test_images"

PREDICTION_DIR = "predictions"
os.makedirs(PREDICTION_DIR, exist_ok=True)

# Global flags for version compatibility (Added to fix GitHub Actions Error)
is_legacy_format = False
model_inference_func = None

# ===============================
# 🔽 DOWNLOAD TEST IMAGES
# ===============================
def download_test_images():
    print("⬇️ Downloading latest test images from Google Drive...")
    gdown.download_folder(url=TEST_FOLDER_URL, output=TEST_DIR, quiet=False, use_cookies=False)
    print("✅ Test images downloaded!")

# ===============================
# 🧩 LOAD MODEL & CLASSES
# ===============================
def download_file(url, output_path):
    print(f"⬇️ Downloading file from: {url}")
    gdown.download(url, output_path, quiet=False)
    print(f"✅ Downloaded: {output_path}")

def load_model_or_exit(path):
    global is_legacy_format, model_inference_func
    if not os.path.exists(path):
        print(f"⚠️ Error: Model directory '{path}' not found.")
        print("Please ensure the 'model_saved' folder is in your GitHub repository.")
        exit(1)
        
    print(f"📦 Loading model from: {path}")
    try:
        # Standard loading for Keras 2 or .keras files
        model = tf.keras.models.load_model(path)
        is_legacy_format = False
        print("✅ Model loaded successfully via standard Keras method!")
        return model
    except Exception as error_msg:
        # Hybrid Mode: Compatibility for Keras 3 with legacy SavedModel folders
        print(f"⚠️ Standard load failed (Keras 3 detected): {str(error_msg)}")
        print("🔄 Switching to TensorFlow SavedModel signature loader...")
        model = tf.saved_model.load(path)
        model_inference_func = model.signatures["serving_default"]
        is_legacy_format = True
        print("✅ Model loaded successfully via TensorFlow Signature mode!")
        return model

def load_class_map():
    if not os.path.exists(CLASS_MAP_FILENAME):
        print("⚠️ Class map not found. Downloading...")
        gdown.download(CLASS_MAP_URL, CLASS_MAP_FILENAME, quiet=False)
        print("✅ Class map downloaded:", CLASS_MAP_FILENAME)

    with open(CLASS_MAP_FILENAME, "r") as f:
        class_indices = json.load(f)
    
    # Mapping indices to names
    return {v: k for k, v in class_indices.items()}

# ===============================
# 🔍 PREDICTION FUNCTION
# ===============================
def predict_image(model, img_path, index_to_class):
    global is_legacy_format, model_inference_func
    
    # Load and preprocess image (Standard 224x224 for MobileNetV2)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    if not is_legacy_format:
        # Keras 2 / Normal Prediction
        preds = model.predict(x)
    else:
        # Keras 3 / Legacy SavedModel Signature Prediction
        input_data = tf.constant(x, dtype=tf.float32)
        inference_results = model_inference_func(input_data)
        # Getting the tensor from the dictionary output
        preds = list(inference_results.values())[0].numpy()

    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    
    # Convert index to class name
    if index_to_class:
        class_name = index_to_class[class_id]
    else:
        class_name = str(class_id)

    return class_id, class_name, confidence

# ===============================
# 🚀 RUN BATCH PREDICTION WITH VALIDATION
# ===============================
def run_batch():
    print(f"🕐 Starting batch run: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    download_test_images()

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"No directory named '{TEST_DIR}' found.")

    # Loading critical assets
    model = load_model_or_exit(MODEL_DIR)
    index_to_class = load_class_map()

    csv_file = os.path.join(PREDICTION_DIR, "predictions.csv")
    
    # Final Research Columns Structure
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
    
    # Processing each image in the test directory
    for fname in os.listdir(TEST_DIR):
        fpath = os.path.join(TEST_DIR, fname)
        
        # Filter for image files only
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # Regex Extraction: object_index_name_id (Example: watch_1_Amir_11111.png)
        # Supports both simple and complex filename patterns
        match = re.match(r"([A-Za-z]+)(?:_\d+)?_([A-Za-z]+)_(\d+)", fname)
        
        if match:
            true_label = match.group(1).lower()
            cust_name = match.group(2)
            cust_id = match.group(3)
        else:
            true_label = "unknown"
            cust_name = fname
            cust_id = "0"

        # Model Prediction
        class_id, class_name, confidence = predict_image(model, fpath, index_to_class)

        # Accuracy Comparison Logic
        is_match = true_label in class_name.lower()

        # Check if record already exists (Avoid Duplicate Entries)
        existing = df_existing[
            (df_existing["filename"] == fname) &
            (df_existing["predicted_class_name"] == class_name) &
            (np.isclose(df_existing["confidence"].astype(float), confidence, atol=1e-4))
        ]

        if existing.empty:
            # Create a new dictionary record
            new_entry = {
                "filename": fname,
                "true_label": true_label,
                "predicted_class_name": class_name,
                "is_match": is_match,
                "customer_name": cust_name,
                "customer_id": cust_id,
                "confidence": round(confidence, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
            new_records.append(new_entry)
            print(f"🖼️ New Record: {fname} -> AI: {class_name} | Correct: {is_match}")
        else:
            print(f"ℹ️ Skipping existing entry: {fname}")

    # Save results to CSV if there are new records
    if new_records:
        df_new = pd.DataFrame(new_records)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Ensure correct column order before saving
        df_combined = df_combined[columns]
        df_combined.to_csv(csv_file, index=False)
        print(f"✅ CSV database updated: {csv_file}")
    else:
        print("⚠️ No new unique images were found to process.")

# ===============================
# 🏁 MAIN ENTRY POINT
# ===============================
if __name__ == "__main__":
    print(">>> Automated Batch Prediction System (v6.2 - Hybrid Compatibility) <<<")
    run_batch()
    print(">>> Batch Processing Completed Successfully <<<")