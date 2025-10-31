import os
import requests
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

MODEL_URL = "https://drive.google.com/uc?id=1QgatECmfFze9sSoqbuIo1IDwmU1csCXX"
MODEL_FILENAME = "model.h5"
TEST_DIR = "test_images"
OUTPUT_CSV = "daily_predictions.csv"

# 🔹 دانلود مدل در صورت عدم وجود
def download_model():
    if not os.path.exists(MODEL_FILENAME):
        print(f"⬇️ Downloading model from: {MODEL_URL}")
        r = requests.get(MODEL_URL)
        with open(MODEL_FILENAME, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded successfully! File size:", os.path.getsize(MODEL_FILENAME), "bytes")

# 🔹 بارگذاری مدل
def load_model_or_exit(path):
    try:
        model = tf.keras.models.load_model(path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print("❌ Failed to load model:", e)
        exit(1)

# 🔹 خواندن لیست کلاس‌ها
def get_class_names():
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ No directory named '{TEST_DIR}' found.")
        exit(1)
    class_names = sorted(os.listdir(TEST_DIR))
    print(f"📂 Found {len(class_names)} classes: {class_names[:10]} ...")
    return class_names

# 🔹 پیش‌بینی برای یک تصویر
def predict_image(model, img_path, class_names):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Unknown({class_idx})"
    confidence = float(np.max(preds[0]))
    return class_name, confidence

# 🔹 ذخیره نتایج در CSV
def save_predictions(predictions):
    df = pd.DataFrame(predictions)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"📄 Saved predictions to {OUTPUT_CSV}")

# 🔹 اجرای Batch Prediction
def run_batch():
    print(">>> Running UPDATED main.py version 1.3 <<<")
    print(f"شروع اجرای batch: {datetime.utcnow().isoformat()} UTC")

    download_model()
    model = load_model_or_exit(MODEL_FILENAME)
    class_names = get_class_names()

    predictions = []
    for cls in class_names:
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue

        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(cls_dir, fname)
            pred_label, conf = predict_image(model, fpath, class_names)
            predictions.append({
                "timestamp": datetime.utcnow().isoformat(),
                "true_class": cls,
                "file_name": fname,
                "predicted_class": pred_label,
                "confidence": round(conf, 4)
            })

    save_predictions(predictions)
    print(f"✅ Batch completed — {len(predictions)} images processed.")

if __name__ == "__main__":
    run_batch()
