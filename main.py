import tensorflow as tf
import numpy as np
import os

import requests
from PIL import Image
from datetime import datetime
MODEL_URL = "https://drive.google.com/uc?id=1QgatECmfFze9sSoqbuIo1IDwmU1csCXX"
MODEL_FILENAME = "model.h5"
TEST_IMAGES_DIR = "test_images"
OUTPUT_CSV = "daily_predictions.csv"

print(">>> Running UPDATED main.py version 1.2 <<<")

def download_model():
    """دانلود مدل از Google Drive"""
    if os.path.exists(MODEL_FILENAME):
        print(f"✅ Model file already exists: {MODEL_FILENAME}")
        return MODEL_FILENAME

    print(f"⬇️ Downloading model from: {MODEL_URL}")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_FILENAME, "wb") as f:
            f.write(response.content)
        print(f"✅ Model downloaded successfully! File size: {os.path.getsize(MODEL_FILENAME)} bytes")
    else:
        print(f"❌ Failed to download model. HTTP {response.status_code}")
        exit(1)
    return MODEL_FILENAME


def load_model_or_exit(path):
    """لود مدل، با بررسی خطا"""
    try:
        print(f"📦 Loading model from {path} ...")
        model = tf.keras.models.load_model(path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array


def predict_images(model):
    """پیش‌بینی روی تصاویر موجود در پوشه test_images"""
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"⚠️ No directory named '{TEST_IMAGES_DIR}' found.")
        return []

    predictions = []
    for filename in os.listdir(TEST_IMAGES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(TEST_IMAGES_DIR, filename)
            img_array = preprocess_image(img_path)
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds)
            predictions.append((filename, int(predicted_class)))
            print(f"🖼 {filename} → class {predicted_class}")
    return predictions


def save_predictions(predictions):
    """ذخیره خروجی پیش‌بینی‌ها در CSV"""
    import pandas as pd
    if predictions:
        df = pd.DataFrame(predictions, columns=["image", "predicted_class"])
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Predictions saved to {OUTPUT_CSV}")
    else:
        print("⚠️ No predictions to save.")


def run_batch():
    print(f"شروع اجرای batch: {datetime.utcnow().isoformat()} UTC")

    model_path = download_model()
    print(f"🔍 Checking downloaded file size: {os.path.getsize(model_path)} bytes")

    model = load_model_or_exit(model_path)

    predictions = predict_images(model)
    save_predictions(predictions)

    print("🏁 Batch completed successfully.")


if __name__ == "__main__":
    run_batch()
