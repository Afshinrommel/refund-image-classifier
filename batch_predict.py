import os
import json
import requests

# ===============================
# 🔧 CONFIGURATION
# ===============================
API_URL = "http://127.0.0.1:5000/predict"  # Flask API endpoint
IMAGES_FOLDER = r"C:/Un/Semster 4/Book/model to production/refund-image-classifier/test_images/"
OUTPUT_JSON = "batch_predictions.json"

# ===============================
# 🚀 GET IMAGE FILES
# ===============================
image_files = [
    f for f in os.listdir(IMAGES_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

results = []
success_count = 0
failure_count = 0

# ===============================
# 📤 SEND IMAGES TO API AND COLLECT PREDICTIONS
# ===============================
for img_file in image_files:
    img_path = os.path.join(IMAGES_FOLDER, img_file)
    
    # Optional: rename to safe filename for API
    safe_filename = img_file.replace(" ", "_").replace("(", "").replace(")", "")
    
    with open(img_path, "rb") as f:
        files = {"file": (safe_filename, f)}
        try:
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
            prediction = response.json()
            results.append({"image": img_file, **prediction})
            success_count += 1
            print(f"✅ {img_file}: {prediction}")

        except requests.exceptions.RequestException as e:
            print(f"❌ {img_file}: Request failed: {e}")
            results.append({"image": img_file, "error": str(e)})
            failure_count += 1

        except ValueError:
            print(f"❌ {img_file}: Invalid response from server: {response.text}")
            results.append({"image": img_file, "error": response.text})
            failure_count += 1

# ===============================
# 📄 SAVE RESULTS TO JSON
# ===============================
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

# ===============================
# 📝 SUMMARY
# ===============================
total_images = len(image_files)
print(f"\n✅ Batch predictions completed: {total_images} images processed")
print(f"✅ Successes: {success_count}")
print(f"❌ Failures: {failure_count}")
print(f"📄 Results saved to '{OUTPUT_JSON}'")
