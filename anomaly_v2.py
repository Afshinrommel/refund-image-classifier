# -*- coding: utf-8 -*-
"""Caltech101_Final_Trainer.ipynb"""

# STEP 1️⃣ — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 2️⃣ — Install requirements
!pip install -q kaggle tensorflow matplotlib scikit-learn pandas

# STEP 3️⃣ — Upload kaggle.json (manually)
from google.colab import files
files.upload()  # Upload your kaggle.json file

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# STEP 4️⃣ — Download and extract Caltech-101
!kaggle datasets download -d imbikramsaha/caltech-101 -p /content/
!unzip -q /content/caltech-101.zip -d /content/dataset

# STEP 5️⃣ — Split dataset (train/validation)
import os, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

dataset_path = "/content/dataset/caltech-101"
base_dir = Path("/content/caltech_split")
train_dir = base_dir / "train"
val_dir = base_dir / "val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for category in os.listdir(dataset_path):
    cat_path = os.path.join(dataset_path, category)
    if os.path.isdir(cat_path):
        images = [os.path.join(cat_path, f) for f in os.listdir(cat_path) if f.endswith('.jpg')]
        if len(images) < 5:  # skip small folders
            continue
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        os.makedirs(train_dir / category, exist_ok=True)
        os.makedirs(val_dir / category, exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, train_dir / category)
        for img in val_imgs:
            shutil.copy(img, val_dir / category)

print("✅ Dataset split complete.")

# STEP 6️⃣ — Build and train model with fine-tuning
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=25, width_shift_range=0.2,
                                   height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# 🔹 Load MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# 🔹 Freeze most layers, fine-tune last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

# 🔹 Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

# 🔹 Train model (increased epochs)
history = model.fit(train_gen, validation_data=val_gen, epochs=25)
print("✅ Model training complete!")

# STEP 7️⃣ — Save model and class mapping
import json

model_path = "/content/drive/MyDrive/caltech_model_finetuned.h5"
mapping_path = "/content/drive/MyDrive/class_indices.json"

model.save(model_path)
with open(mapping_path, "w") as f:
    json.dump(train_gen.class_indices, f)

print("💾 Model and class mapping saved to Google Drive.")

# STEP 8️⃣ — Evaluate and visualize
import matplotlib.pyplot as plt

acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# STEP 9️⃣ — Test sample predictions
import numpy as np

labels = list(train_gen.class_indices.keys())
x_batch, y_batch = next(val_gen)
preds = model.predict(x_batch)
pred_classes = np.argmax(preds, axis=1)

plt.figure(figsize=(12,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_batch[i])
    plt.title(f"Pred: {labels[pred_classes[i]]}")
    plt.axis("off")
plt.show()

print("✅ All steps completed successfully.")
