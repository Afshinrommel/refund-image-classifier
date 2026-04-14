import os
# Force Keras 2 legacy behavior for compatibility with TF 2.19
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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
# Direct link to the model (.h5) from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1fA5v1y4IM81dbOqOzNOfidjgfViPIizs"
MODEL_PATH = "final_model_caltech.h5"

# Direct link to the class mapping (class_indices.json)
CLASS_MAP_URL = "https://drive.google.com/uc?id=1uqqVbCuodTN6Nbc-isDhwnDjUqhWZTKc"
CLASS_MAP_FILENAME = "class_indices.json"

# Link to the test images folder in Google Drive
TEST_FOLDER_URL = "https://drive.google.com/drive/folders/1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc?usp=drive_link"
TEST_DIR = "test_images"

PREDICTION_DIR = "predictions"
os.makedirs(PREDICTION_DIR, exist_ok=True)

# ===============================
# DOWNLOAD FUNCTIONS
# ===============================
def download_assets():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    if not os.path.exists(CLASS_MAP_FILENAME):
        print("Downloading class map from Google Drive...")
        gdown.download(CLASS_MAP_URL, CLASS_MAP_FILENAME, quiet=False)

    print("Downloading test images...")
    gdown.download_folder(url=TEST_FOLDER_URL, output=TEST_DIR, quiet=False, use_cookies=False)

def load_class_map():
    with open(CLASS_MAP_FILENAME, "r") as f:
        class_indices = json.load(f)
    # Convert numeric index to string label
    return {int(v): k for k, v in class_indices.items()}

# ===============================
# MODEL LOADING
# ===============================
def load_model_safely():
    print(f"Loading H5 model: {MODEL_PATH}")
    try:
        # Load the model without compilation for inference
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("SUCCESS: Model loaded successfully!")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR during model load: {str(e)}")
        exit(1)

# ===============================
# PREDICTION LOGIC
# ===============================
def predict_image(model, img_path, index_to_class):
    # Automatically resize images to 224x224 as required by MobileNetV2
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    class_name = index_to_class.get(class_id, f"Unknown({class_id})")
    return class_name, confidence

def run_batch():
    download_assets()
    model = load_model_safely()
    index_to_class = load_class_map()

    # Synonyms dictionary to handle naming variations
    aliases = {
        "mandolin": "guitar",
        "guitar": "mandolin"
    }

    csv_file = os.path.join(PREDICTION_DIR, "predictions.csv")
    new_records = []
    
    # Iterate through all images in the test folder
    for fname in os.listdir(TEST_DIR):
        fpath = os.path.join(TEST_DIR, fname)
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # Extract true label from filename (part before first underscore/dash)
        true_label_raw = re.split(r'[_|-]', fname)[0].lower()
        
        predicted_class, confidence = predict_image(model, fpath, index_to_class)
        pred_lower = predicted_class.lower()
        
        # Core validation logic including aliases
        is_correct = (
            true_label_raw in pred_lower or 
            pred_lower in true_label_raw or
            aliases.get(true_label_raw) == pred_lower
        )

        new_records.append({
            "filename": fname,
            "true_label": true_label_raw,
            "predicted_class_name": predicted_class,
            "is_correct": "YES" if is_correct else "NO",
            "confidence": round(confidence, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        })
        print(f"Processed: {fname} -> Result: {'YES' if is_correct else 'NO'}")

    if new_records:
        df = pd.DataFrame(new_records)
        df.to_csv(csv_file, index=False)
        print(f"All results updated in {csv_file}")
    else:
        print("No images found to process.")

if __name__ == "__main__":
    run_batch()