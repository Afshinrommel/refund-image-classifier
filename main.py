import tensorflow as tf
import numpy as np
import os
import sys
import requests
from datetime import datetime
from PIL import Image

MODEL_FILENAME = "model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1QgatECmfFze9sSoqbuIo1IDwmU1csCXX"
IMAGE_DIR = "images"
OUTPUT_CSV = "daily_predictions.csv"


def download_model_if_needed():
    """دانلود مدل از گوگل درایو اگر موجود نباشد."""
    if os.path.exists(MODEL_FILENAME):
        print(f"✅ مدل '{MODEL_FILENAME}' از قبل وجود دارد.")
        return
    print(f"⬇️ در حال دانلود مدل از Google Drive ...")
    try:
        r = requests.get(MODEL_URL)
        if r.status_code != 200:
            print(f"❌ خطا در دانلود مدل: وضعیت HTTP {r.status_code}")
            sys.exit(1)
        with open(MODEL_FILENAME, "wb") as f:
            f.write(r.content)
        print("✅ دانلود مدل انجام شد.")

        # بررسی اندازه فایل
        size = os.path.getsize(MODEL_FILENAME)
        print(f"📦 اندازه فایل مدل: {size} بایت")
        if size < 10000:  # کمتر از 10 کیلوبایت = فایل خراب
            print("❌ فایل مدل به نظر می‌رسد خراب یا ناقص است.")
            sys.exit(1)

    except Exception as e:
        print("❌ خطا در دانلود مدل:", e)
        sys.exit(1)


def load_model_or_exit(path):
    """بارگذاری مدل با مدیریت خطا."""
    try:
        print(f"📂 در حال بارگذاری مدل از '{path}' ...")
        model = tf.keras.models.load_model(path)
        print("✅ مدل با موفقیت بارگذاری شد.")
        return model
    except Exception as e:
        print(f"❌ خطا در بارگذاری مدل: {e}")
        sys.exit(1)


def preprocess_image(image_path, target_size=(224, 224)):
    """پیش‌پردازش تصویر برای مدل."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_image(model, image_path):
    """پیش‌بینی کلاس تصویر."""
    img = preprocess_image(image_path)
    preds = model.predict(img)
    return preds


def run_batch():
    print("🚀 شروع اجرای batch:", datetime.utcnow().isoformat(), "UTC")

    download_model_if_needed()
    model = load_model_or_exit(MODEL_FILENAME)

    if not os.path.exists(IMAGE_DIR):
        print(f"❌ پوشه تصاویر '{IMAGE_DIR}' یافت نشد.")
        sys.exit(1)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print("⚠️ هیچ تصویری برای پردازش یافت نشد.")
        sys.exit(0)

    print(f"📸 تعداد تصاویر: {len(image_files)}")

    results = []
    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        preds = predict_image(model, img_path)
        predicted_class = np.argmax(preds)
        confidence = float(np.max(preds))
        results.append((img_name, predicted_class, confidence))
        print(f"✅ {img_name}: کلاس {predicted_class} (اطمینان {confidence:.2f})")

    # ذخیره در CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class", "confidence"])
        writer.writerows(results)

    print(f"💾 نتایج در فایل '{OUTPUT_CSV}' ذخیره شدند.")


if __name__ == "__main__":
    run_batch()
