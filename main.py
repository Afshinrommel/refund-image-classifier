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
# Specific path to your new .keras model file
MODEL_DIR = os.path.join("model_saved", "final_model_caltech.keras")

CLASS_MAP_URL = "https://drive.google.com/uc?id=1RL6oiB8UOVkOmCsjqO5qbJsS0g7ZFMqf"
CLASS_MAP_FILENAME = "class_indices.json"

TEST_FOLDER_URL = "https://drive.google.com/drive/folders/1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc?usp=drive_link"
TEST_DIR = "test_images"

PREDICTION_DIR = "predictions"
os.makedirs(PREDICTION_DIR, exist_ok=True)

# Global flags for Keras 3 / TensorFlow version compatibility
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
    
    # Priority: Check if the specific .keras file exists
    if not os.path.exists(path):
        print(f"⚠️ Specific file '{path}' not found. Searching for directory fallback...")
        path = "model_saved" # Fallback to the folder if file is missing
        if not os.path.exists(path):
            print(f"❌ Error: Model path '{path}' not found. Please check your GitHub repository.")
            exit(1)
        
    print(f"📦 Loading model from: {path}")
    try:
        # Preferred method: Loading the .keras file (Standard Keras 3)
        model = tf.keras.models.load_model(path)
        is_legacy_format = False
        print(f"✅ SUCCESS: Loaded '{os.path.basename(path)}' via standard Keras method!")
        return model
    except Exception as error_msg:
        # Fallback method: Loading legacy SavedModel folders via TF Signatures
        print(f"ℹ️ Standard load failed or not a .keras file. Switch to Hybrid Mode.")
        print(f"Error details: {str(error_msg)}")
        model = tf.saved_model.load(path)
        model_inference_func = model.signatures["serving_default"]
        is_legacy_format = True
        print("✅ SUCCESS: Loaded via TensorFlow Signature mode!")
        return model

def load_class_map():
    if not os.path.exists(CLASS_MAP_FILENAME):
        print("⚠️ Class map not found. Downloading...")
        gdown.download(CLASS_MAP_URL, CLASS_MAP_FILENAME, quiet=False)
        print("✅ Class map downloaded:", CLASS_MAP_FILENAME)

    with open(CLASS_MAP_FILENAME, "r") as f:
        class_indices = json.load(f)
    
    # Convert index (int) to class name (string)
    return {v: k for k, v in class_indices.items()}

# ===============================
# 🔍 PREDICTION FUNCTION
# ===============================
def predict_image(model, img_path, index_to_class):
    global is_legacy_format, model_inference_func
    
    # Image preprocessing for MobileNetV2
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    if not is_legacy_format:
        # Normal prediction for .keras files
        preds = model.predict(x)
    else:
        # Dictionary-based prediction for SavedModel folders
        input_data = tf.constant(x, dtype=tf.float32)
        inference_results = model_inference_func(input_data)
        preds = list(inference_results.values())[0].numpy()

    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    class_name = index_to_class[class_id] if index_to_class else str(class_id)

    return class_id, class_name, confidence

# ===============================
# 🚀 RUN BATCH PREDICTION WITH VALIDATION
# ===============================
def run_batch():
    print(f"🕐 Starting batch run: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    download_test_images()

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Directory '{TEST_DIR}' not found.")

    # Assets Loading
    model = load_model_or_exit(MODEL_DIR)
    index_to_class = load_class_map()

    csv_file = os.path.join(PREDICTION_DIR, "predictions.csv")
    
    # 8 Columns as strictly requested
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

        # Regex: Extracting true_label, customer_name, and customer_id
        # Pattern: [object]_[index]_[name]_[id].extension
        match = re.match(r"([A-Za-z]+)(?:_\d+)?_([A-Za-z]+)_(\d+)", fname)
        
        if match:
            true_label = match.group(1).lower()
            cust_name = match.group(2)
            cust_id = match.group(3)
        else:
            true_label = "unknown"
            cust_name = fname
            cust_id = "0"

        # AI Analysis
        class_id, class_name, confidence = predict_image(model, fpath, index_to_class)

        # Scientific Accuracy Check
        is_match = true_label in class_name.lower()

        # Duplicate Avoidance Logic (Checking filename, AI result, and confidence)
        existing = df_existing[
            (df_existing["filename"] == fname) &
            (df_existing["predicted_class_name"] == class_name) &
            (np.isclose(df_existing["confidence"].astype(float), confidence, atol=1e-4))
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
            print(f"ℹ️ Entry already exists: {fname}")

    # Finalize CSV
    if new_records:
        df_new = pd.DataFrame(new_records)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined[columns]
        df_combined.to_csv(csv_file, index=False)
        print(f"✅ CSV updated at: {csv_file}")
    else:
        print("⚠️ No new records to save.")

# ===============================
# 🏁 MAIN ENTRY POINT
# ===============================
if __name__ == "__main__":
    print(">>> Automated Batch Prediction System (v7.0 - Precision Mode) <<<")
    run_batch()
    print(">>> Batch Processing Completed Successfully <<<")