import os
import json
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, redirect, url_for, session, render_template_string
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

# ======================================================
# CONFIGURATION
# ======================================================
import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

MODEL_PATH = "model_saved"
CLASS_MAP_FILENAME = "class_indices.json"
CLIENT_SECRET_FILE = "client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
UPLOAD_FOLDER = "uploads"
DRIVE_FOLDER_ID = "1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc"  # your shared folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask setup
app = Flask(__name__)
app.secret_key = "replace_this_with_a_secure_key"

# ======================================================
# LOAD MODEL (unchanged)
# ======================================================
print("🔄 Loading model from:", MODEL_PATH)
model = None
infer = None
is_keras_model = False
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded via tf.keras.models.load_model")
    if not hasattr(model, "predict"):
        model = tf.saved_model.load(MODEL_PATH)
        infer = model.signatures["serving_default"]
    else:
        is_keras_model = True
except Exception as e:
    print(f"⚠️ Model load failed: {e}")
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serving_default"]

print("✅ Model ready!")

# ======================================================
# LOAD CLASS MAP
# ======================================================
if os.path.exists(CLASS_MAP_FILENAME):
    with open(CLASS_MAP_FILENAME) as f:
        index_to_class = {v: k for k, v in json.load(f).items()}
else:
    index_to_class = None
    print("⚠️ No class map found — will return numeric labels")

# ======================================================
# GOOGLE OAUTH2 AUTHORIZATION
# ======================================================
#def get_flow():
#    return Flow.from_client_secrets_file(
#        CLIENT_SECRET_FILE,
#        scopes=SCOPES,
#        redirect_uri=url_for("oauth2callback", _external=True),
#    )

def get_flow():
    return Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri="http://localhost:5000/oauth2callback",  # ✅ force localhost
    )


@app.route("/authorize")
def authorize():
    flow = get_flow()
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    return redirect(auth_url)


@app.route("/oauth2callback")
def oauth2callback():
    flow = get_flow()
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials
    session["credentials"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    return redirect(url_for("upload_page"))

# ======================================================
# UPLOAD FILE TO GOOGLE DRIVE
# ======================================================
def upload_to_drive(filepath, filename):
    if "credentials" not in session:
        return {"error": "User not authorized. Please visit /authorize first."}

    creds_data = session["credentials"]
    creds = Credentials(**creds_data)
    service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": filename, "parents": [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(filepath, resumable=True)

    uploaded_file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id, webViewLink")
        .execute()
    )

    return {
        "id": uploaded_file["id"],
        "link": uploaded_file["webViewLink"],
    }

# ======================================================
# FLASK ROUTES
# ======================================================
@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("upload_page"))

@app.route("/upload", methods=["GET"])
def upload_page():
    return """
    <html>
    <body style="font-family:sans-serif; text-align:center;">
        <h2>Refund Image Classifier (Local)</h2>
        <form id="form" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required><br><br>
            <button type="button" onclick="predict()">Predict</button>
            <button type="button" onclick="uploadToDrive()">Upload to Drive</button>
        </form>
        <pre id="result"></pre>
        <p><a href="/authorize">Authorize Google Drive</a></p>
        <script>
            async function predict() {
                const formData = new FormData(document.getElementById('form'));
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('result').innerText = JSON.stringify(data, null, 2);
            }
            async function uploadToDrive() {
                const formData = new FormData(document.getElementById('form'));
                const res = await fetch('/upload_drive_only', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('result').innerText = JSON.stringify(data, null, 2);
                if (data.link && data.link.startsWith("http"))
                    window.open(data.link, "_blank");
            }
        </script>
    </body>
    </html>
    """

# ======================================================
# UPLOAD DRIVE ONLY (REPLACES OLD DRIVE CODE)
# ======================================================
@app.route("/upload_drive_only", methods=["POST"])
def upload_drive_only():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(img_file.filename))
    img_file.save(filepath)

    try:
        result = upload_to_drive(filepath, img_file.filename)
    except Exception as e:
        result = {"error": str(e)}
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)

# ======================================================
# PREDICT (UNTOUCHED)
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    local_path = os.path.join(UPLOAD_FOLDER, secure_filename(img_file.filename))
    img_file.save(local_path)

    try:
        img = image.load_img(local_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        if is_keras_model:
            preds = model.predict(x)
        else:
            outputs = infer(tf.constant(x, dtype=tf.float32))
            preds = list(outputs.values())[0].numpy()

        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        class_name = index_to_class[class_id] if index_to_class else str(class_id)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass

    return jsonify({
        "predicted_class": class_name,
        "confidence": confidence
    })

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
