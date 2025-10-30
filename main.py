#!/usr/bin/env python3
"""
main.py
اجرای batch prediction برای تصاویر جدید در caltech_split/val/
خروجی: daily_predictions.csv (حالت append)
توضیحات کامل را در README.md ببینید.
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

# تذکر: tensorflow هنگام اجرای workflow نصب می‌شود؛ اگر اجرا محلی می‌کنید، اطمینان حاصل کنید tensorflow نصب باشد.
try:
    import tensorflow as tf
except Exception as e:
    raise SystemExit("خطا: کتابخانه tensorflow پیدا نشد. لطفا آن را نصب کنید. " + str(e))

MODEL_FILENAME = "caltech_model.h5"   # مدل باید در ریشه ریپازیتوری قرار بگیرد
DATA_DIR = "caltech_split/val"
OUTPUT_CSV = "daily_predictions.csv"

IMG_SIZE = (224, 224)

def load_model_or_exit(path):
    if not os.path.exists(path):
        raise SystemExit(f"مدل یافت نشد: {path}. لطفا فایل '{path}' را در ریشه ریپازیتوری قرار دهید.")
    model = tf.keras.models.load_model(path)
    return model

def list_classes(base_dir):
    # فقط پوشه‌های سطح اول را به عنوان کلاس در نظر می‌گیریم
    classes = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    return classes

def predict_image(model, img_path, class_names):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    pred_idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    return pred_label, conf

def run_batch():
    print("شروع اجرای batch:", datetime.utcnow().isoformat(), "UTC")
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"پوشه داده پیدا نشد: {DATA_DIR}. ساختار مورد انتظار: {DATA_DIR}/<class_name>/*.jpg")

    model = load_model_or_exit(MODEL_FILENAME)
    class_names = list_classes(DATA_DIR)
    if len(class_names) == 0:
        raise SystemExit("هیچ کلاس (زیرپوشه‌ای) در مسیر داده پیدا نشد.")

    processed_files = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            processed_files = set(existing['file_name'].astype(str).tolist())
        except Exception:
            processed_files = set()

    rows = []
    for cls in class_names:
        cls_path = os.path.join(DATA_DIR, cls)
        for fname in sorted(os.listdir(cls_path)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if fname in processed_files:
                continue
            fpath = os.path.join(cls_path, fname)
            try:
                pred_label, conf = predict_image(model, fpath, class_names)
            except Exception as e:
                print(f"خطا هنگام پیش‌بینی {fpath}: {e}")
                continue
            rows.append({
                "timestamp": datetime.utcnow().isoformat(),
                "true_class": cls,
                "file_name": fname,
                "predicted_class": pred_label,
                "confidence": conf
            })

    if rows:
        df_new = pd.DataFrame(rows)
        if os.path.exists(OUTPUT_CSV):
            df_old = pd.read_csv(OUTPUT_CSV)
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ پردازش شد: {len(rows)} تصویر جدید. نتایج در '{OUTPUT_CSV}' ذخیره شد (حالت append).")
    else:
        print("ℹ️ هیچ تصویر جدیدی برای پردازش یافت نشد.")

if __name__ == "__main__":
    run_batch()
