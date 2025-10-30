import os
import requests
import tensorflow as tf
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing import image
import numpy as np

# ========================
# تنظیمات پایه
# ========================
MODEL_URL = "https://drive.usercontent.google.com/download?id=1QgatECmfFze9sSoqbuIo1IDwmU1csCXX&export=download"
MODEL_FILENAME = "model.h5"
IMAGE_FOLDER = "images"
OUTPUT_CSV = "daily_predictions.csv"

# ========================
# تابع دانلود مدل
# ========================
def download_model():
    """دانلود مدل از Google Drive با بررسی صحت فایل"""
    if os.path.exists(MODEL_FILENAME):
        print(f"✅ Model file '{MODEL_FILENAME}' already exists.")
        return

    print("📥 Downloading model from Google Drive...")
    response = requests.get(MODEL_URL, allow_redirects=True)
    if response.status_code == 200 and b"HDF" in response.content[:4]:
        with open(MODEL_FILENAME, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully.")
    else:
        print("❌ Model download failed or invalid file received.")
        print("Response header:", response.headers)
        print("Response content (first 200 bytes):", response.content[:200])
        exit(1)

# ========================
# بارگذاری مدل
# ========================
def load_model_or_exit(path):
    """بارگذاری مدل از فایل"""
    try:
        model = tf.keras.models.load_model(path)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print("❌ Error loading model:", e)
        exit(1)

# ========================
# آماده‌سازی تصویر برای مدل
# ========================
def prepare_image(img_path, target_size=(224, 224)):
    """تبدیل تصویر به آرایه‌ی مناسب برای مدل"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ========================
# اجرای پیش‌بینی روی تصاویر پوشه
# ========================
def run_batch():
    print(f"شروع اجرای batch: {datetime.utcnow().isoformat()} UTC")

    # دانلود مدل اگر وجود ندارد
    download_model()

    # بارگذاری مدل
    model = load_model_or_exit(MODEL_FILENAME)

    # بررسی وجود فولدر تصاویر
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Folder '{IMAGE_FOLDER}' not found!")
        exit(1)

    results = []
    class_names = list(model.output_names) if model.output_names else [str(i) for i in range(model.output_shape[-1])]

    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(IMAGE_FOLDER, filename)
            img_array = prepare_image(img_path)
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            results.append({
                "filename": filename,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
            print(f"🖼️ {filename} → Class {predicted_class} ({confidence:.2f})")

    if not results:
        print("⚠️ No images found for prediction.")
        exit(0)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Results saved to '{OUTPUT_CSV}'")

# ========================
# اجرای مستقیم فایل
# ========================
if __name__ == "__main__":
    run_batch()
