import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import datetime
import requests

# -----------------------------
# تنظیمات اولیه
# -----------------------------
MODEL_FILENAME = "caltech_model.h5"
#MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # در صورت نیاز جایگزین کن
#MODEL_URL = "https://drive.google.com/file/d/1QgatECmfFze9sSoqbuIo1IDwmU1csCXX/view?usp=sharing"
#MODEL_URL = "https://drive.google.com/uc?id=1QgatECmfFze9sSoqbuIo1IDwmU1csCXX"
MODEL_URL = "https://drive.usercontent.google.com/download?id=1QgatECmfFze9sSoqbuIo1IDwmU1csCXX&export=download"

IMAGE_DIR = "caltech_split/val"
OUTPUT_FILE = "daily_predictions.csv"

# -----------------------------
# تابع بارگذاری یا ساخت مدل
# -----------------------------
def load_or_build_model():
    if os.path.exists(MODEL_FILENAME):
        try:
            print(f"Loading model from {MODEL_FILENAME} ...")
            model = tf.keras.models.load_model(MODEL_FILENAME)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")

    # اگر مدل وجود ندارد، تلاش برای دانلود از Google Drive
    if MODEL_URL and "YOUR_FILE_ID" not in MODEL_URL:
        print("Downloading model from Google Drive...")
        r = requests.get(MODEL_URL)
        with open(MODEL_FILENAME, "wb") as f:
            f.write(r.content)
        try:
            model = tf.keras.models.load_model(MODEL_FILENAME)
            print("✅ Model downloaded and loaded successfully.")
            return model
        except Exception as e:
            print(f"⚠️ Downloaded file not valid model: {e}")

    # در صورت شکست، مدل موقتی می‌سازد
    print("⚠️ No valid model found. Creating fallback model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

# -----------------------------
# تابع پردازش تصاویر و پیش‌بینی
# -----------------------------
def predict_images(model):
    results = []
    if not os.path.exists(IMAGE_DIR):
        print(f"⚠️ Image directory '{IMAGE_DIR}' not found.")
        return results

    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                try:
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    predictions = model.predict(img_array)
                    pred_class = np.argmax(predictions, axis=1)[0]
                    results.append({
                        "image": img_path,
                        "predicted_class": int(pred_class)
                    })
                except Exception as e:
                    print(f"⚠️ Error processing {file}: {e}")
    return results

# -----------------------------
# تابع اصلی اجرای batch
# -----------------------------
def run_batch():
    print(f"شروع اجرای batch: {datetime.utcnow().isoformat()} UTC")
    model = load_or_build_model()
    results = predict_images(model)

    # ذخیره خروجی
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Prediction results saved to {OUTPUT_FILE}")

# -----------------------------
# اجرای برنامه
# -----------------------------
if __name__ == "__main__":
    run_batch()
