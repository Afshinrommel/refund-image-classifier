import os
import json
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# 🧠 CONFIGURATION SECTION
# ===============================
MODEL_URL = "https://drive.google.com/uc?id=10QVwxFBdz6nhqQoe-LnQSQi87e2kPRLv"
MODEL_FILENAME = "model.h5"
CLASS_MAP_FILENAME = "class_indices.json"
TEST_DIR = "test_images"
OUTPUT_DIR = "predictions"

# ===============================
# 🔽 DOWNLOAD MODEL IF NEEDED
# ===============================
def download_file(url, output_path):
    """دانلود فایل مدل از Google Drive"""
    import gdown
    print(f"⬇️ Downloading model from: {url}")
    gdown.download(url, output_path, quiet=False)
    print(f"✅ Model downloaded successfully: {output_path}")

# ===============================
# 🧩 LOAD MODEL & CLASSES
# ===============================
def load_model_or_exit(path):
    """لود مدل از فایل یا دانلود در صورت نبود"""
    if not os.path.exists(path):
        print("⚠️ Model not found locally. Downloading...")
        download_file(MODEL_URL, path)
    print("📦 Loading model ...")
    model = tf.keras.models.load_model(path)
    print("✅ Model loaded successfully!")
    return model

def load_class_map():
    """لود نگاشت نام کلاس‌ها"""
    if not os.path.exists(CLASS_MAP_FILENAME):
        print("⚠️ class_indices.json not found — classes will show as numbers.")
        return None
    with open(CLASS_MAP_FILENAME, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    print("✅ Loaded class mapping file.")
    return index_to_class

# ===============================
# 🔍 PREDICTION FUNCTION
# ===============================
def predict_image(model, img_path, index_to_class):
    """پیش‌بینی یک تصویر"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    class_name = index_to_class[class_id] if index_to_class else str(class_id)

    return class_id, class_name, confidence

# ===============================
# 🚀 RUN BATCH PREDICTION
# ===============================
def run_batch():
    """اجرای پیش‌بینی دسته‌ای روی تصاویر و ذخیره‌ی نتیجه در فایل CSV با نام زمان‌دار"""
    print(f"🕐 Starting batch run: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"⚠️ No directory named '{TEST_DIR}' found.")

    model = load_model_or_exit(MODEL_FILENAME)
    index_to_class = load_class_map()

    results = []
    for fname in os.listdir(TEST_DIR):
        fpath = os.path.join(TEST_DIR, fname)
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
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
        output_path = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.csv")
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
    else:
        print("⚠️ No images found to process.")

# ===============================
# 🏁 MAIN ENTRY
# ===============================
if __name__ == "__main__":
    print(">>> Running Batch Prediction (v3.0 — Auto Save per Minute) <<<")
    run_batch()
