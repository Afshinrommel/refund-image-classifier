import os
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd
from datetime import datetime

MODEL_URL = "https://drive.google.com/uc?id=1QgatECmfFze9sSoqbuIo1IDwmU1csCXX"
MODEL_FILENAME = "model.h5"
TEST_DIR = "test_images"
OUTPUT_FILE = "daily_predictions.csv"

def download_model():
    print(f"⬇️ Downloading model from: {MODEL_URL}")
    response = requests.get(MODEL_URL)
    with open(MODEL_FILENAME, "wb") as f:
        f.write(response.content)
    size = os.path.getsize(MODEL_FILENAME)
    print(f"✅ Model downloaded successfully! File size: {size} bytes")

def load_model_or_exit(path):
    if not os.path.exists(path):
        download_model()
    print(f"📦 Loading model from {path} ...")
    model = tf.keras.models.load_model(path)
    print("✅ Model loaded successfully!")
    return model

def load_images_from_folder(folder):
    if not os.path.exists(folder):
        print(f"⚠️ No directory named '{folder}' found.")
        return []
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, filename)).resize((224, 224))
            img = np.array(img) / 255.0
            images.append(img)
            filenames.append(filename)
    return np.array(images), filenames

def get_class_names(model, test_dir):
    # سعی می‌کند نام کلاس‌ها را از خود مدل بخواند
    if hasattr(model, "class_names"):
        return model.class_names
    # اگر در مدل نبود، از زیرپوشه‌های test_images می‌گیرد
    elif os.path.isdir(test_dir):
        subdirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        if subdirs:
            print(f"📁 Found class folders: {subdirs}")
            return subdirs
    # در غیر اینصورت عددی برمی‌گرداند
    print("⚠️ Could not detect class names. Using numeric labels instead.")
    return None

def save_predictions(filenames, predictions):
    df = pd.DataFrame({"filename": filenames, "prediction": predictions})
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"📄 Predictions saved to {OUTPUT_FILE}")

def run_batch():
    print(f"شروع اجرای batch: {datetime.utcnow().isoformat()} UTC")
    model = load_model_or_exit(MODEL_FILENAME)
    class_names = get_class_names(model, TEST_DIR)

    images, filenames = load_images_from_folder(TEST_DIR)
    if len(images) == 0:
        print("⚠️ No images found to process.")
        return

    preds = model.predict(images)
    predicted_classes = np.argmax(preds, axis=1)

    if class_names:
        predicted_labels = [class_names[i] for i in predicted_classes]
    else:
        predicted_labels = predicted_classes.tolist()

    save_predictions(filenames, predicted_labels)

if __name__ == "__main__":
    print(">>> Running UPDATED main.py version 1.3 (class names supported) <<<")
    run_batch()
