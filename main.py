import os
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
import gdown

# ===============================
# تنظیمات اصلی
# ===============================
MODEL_FILENAME = "refund_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1QgatECmfFze9sSoqbuIo1IDwmU1csCXX"
INPUT_FOLDER = "input_images"
OUTPUT_CSV = "daily_predictions.csv"

# ===============================
# دانلود مدل در صورت نیاز
# ===============================
def download_model():
    if os.path.exists(MODEL_FILENAME):
        print(f"✅ Model file '{MODEL_FILENAME}' already exists.")
        return

    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError("❌ Failed to download model from Google Drive.")
    print("✅ Model downloaded successfully.")

# ===============================
# بارگذاری مدل
# ===============================
def load_model_or_exit(path):
    try:
        model = tf.keras.models.load_model(path)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise SystemExit(1)

# ===============================
# پیش‌پردازش تصویر
# ===============================
def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ===============================
# اجرای batch
# ===============================
def run_batch():
    start_time = datetime.utcnow().isoformat()
    print(f"شروع اجرای batch: {start_time} UTC")

    # دانلود و بارگذاری مدل
    download_model()
    model = load_model_or_exit(MODEL_FILENAME)

    # بررسی وجود پوشه ورودی
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder '{INPUT_FOLDER}' not found.")
        raise SystemExit(1)

    image_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        print("⚠️ No image files found in input folder.")
        raise SystemExit(1)

    print(f"📸 Found {len(image_files)} images for prediction.")

    # پیش‌بینی
    results = []
    for img_name in image_files:
        img_path = os.path.join(INPUT_FOLDER, img_name)
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array, verbose=0)[0][0]
        results.append({"image": img_name, "prediction": float(prediction)})

    # ذخیره نتایج در CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Predictions saved to '{OUTPUT_CSV}'.")

    end_time = datetime.utcnow().isoformat()
    print(f"⏱ پایان اجرای batch: {end_time} UTC")

# ===============================
# اجرای اصلی
# ===============================
if __name__ == "__main__":
    run_batch()
