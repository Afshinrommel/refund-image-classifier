import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, redirect, url_for, session
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
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

MODEL_PATH = "model_saved"
CLASS_MAP_FILENAME = "class_indices.json"
CLIENT_SECRET_FILE = "client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
UPLOAD_FOLDER = "uploads"
DRIVE_FOLDER_ID = "1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc"
CUSTOMER_FILE = "customers.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask setup
app = Flask(__name__)
app.secret_key = "replace_this_with_a_secure_key"

# ======================================================
# LOAD CUSTOMER DATA
# ======================================================
with open(CUSTOMER_FILE) as f:
    CUSTOMERS = json.load(f)

# ======================================================
# LOAD MODEL
# ======================================================
model = None
infer = None
is_keras_model = False
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    if not hasattr(model, "predict"):
        model = tf.saved_model.load(MODEL_PATH)
        infer = model.signatures["serving_default"]
    else:
        is_keras_model = True
except Exception:
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serving_default"]

# ======================================================
# LOAD CLASS MAP
# ======================================================
if os.path.exists(CLASS_MAP_FILENAME):
    with open(CLASS_MAP_FILENAME) as f:
        index_to_class = {v: k for k, v in json.load(f).items()}
else:
    index_to_class = None

# ======================================================
# GOOGLE OAUTH2
# ======================================================
def get_flow():
    return Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri="http://localhost:5000/oauth2callback"
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
# VALIDATE CUSTOMER
# ======================================================
@app.route("/validate_customer", methods=["POST"])
def validate_customer():
    data = request.get_json()
    if not data:
        return jsonify({"valid": False, "error": "No data provided"}), 400

    name = data.get("name", "").strip().lower()
    cust_id = data.get("id", "").strip()
    valid = any(
        c.get("name", "").strip().lower() == name and c.get("id", "").strip() == cust_id
        for c in CUSTOMERS
    )

    return jsonify({"valid": valid, "name": name if valid else "", "id": cust_id if valid else ""})

# ======================================================
# UPLOAD FILE TO DRIVE
# ======================================================
def upload_to_drive(filepath, filename):
    if "credentials" not in session:
        return {"error": "User not authorized. Please visit /authorize first."}
    creds_data = session["credentials"]
    creds = Credentials(**creds_data)
    service = build("drive", "v3", credentials=creds)
    file_metadata = {"name": filename, "parents": [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(filepath, resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata, media_body=media, fields="id, webViewLink"
    ).execute()
    return {"id": uploaded_file["id"], "link": uploaded_file["webViewLink"]}

# ======================================================
# FRONTEND PAGE
# ======================================================
@app.route("/upload", methods=["GET"])
def upload_page():
    return """
<html>
<head>
<title>Refund Image Classifier</title>
<style>
body { font-family:sans-serif; background:#222; color:#fff; display:flex; justify-content:center; align-items:center; height:100vh; }
.card { background:#333; padding:30px; border-radius:15px; width:400px; text-align:center; }
input, button { margin:5px 0; padding:10px; width:90%; border-radius:8px; border:none; }
button:disabled { background:#555; cursor:not-allowed; }
pre { text-align:left; font-size:12px; background:#111; padding:10px; border-radius:10px; color:#0f0; overflow-x:auto; }
</style>
</head>
<body>
<div class="card">
<h2>🧠 Refund Image Classifier</h2>
<input type="text" id="cust_name" placeholder="Enter Customer Name">
<input type="text" id="cust_id" placeholder="Enter Customer ID">
<button id="validate_btn" onclick="validateCustomer()">✅ Validate Customer</button>
<br>
<input type="file" id="image_file" accept="image/*" disabled>
<button id="predict_btn" onclick="predict()" disabled>🔍 Predict</button>
<button id="upload_btn" onclick="uploadToDrive()" disabled>☁️ Upload to Drive</button>
<pre id="result"></pre>
<p><a href="/authorize">🔑 Authorize Google Drive</a></p>
</div>

<script>
async function validateCustomer() {
    const name = document.getElementById('cust_name').value.trim();
    const id = document.getElementById('cust_id').value.trim();
    if(!name || !id) return alert("Enter name and ID first!");
    
    const res = await fetch('/validate_customer', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name:name,id:id})
    });
    const data = await res.json();
    if(data.valid){
        alert("✅ Customer Validated!");
        document.getElementById('image_file').disabled = false;
        document.getElementById('predict_btn').disabled = false;
        document.getElementById('upload_btn').disabled = false;
    } else {
        alert("❌ Invalid Name or Customer ID");
        document.getElementById('image_file').disabled = true;
        document.getElementById('predict_btn').disabled = true;
        document.getElementById('upload_btn').disabled = true;
    }
}

async function predict() {
    const fileInput = document.getElementById('image_file');
    if(!fileInput.files[0]) return alert("Select an image first!");
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    const res = await fetch('/predict',{method:'POST',body:formData});
    const data = await res.json();
    document.getElementById('result').innerText = JSON.stringify(data,null,2);
}

async function uploadToDrive() {
    const fileInput = document.getElementById('image_file');
    const name = document.getElementById('cust_name').value.trim();
    const id = document.getElementById('cust_id').value.trim();
    if (!fileInput.files[0]) return alert("Select an image first!");
    if (!name || !id) return alert("Customer name and ID missing!");

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    formData.append("name", name);
    formData.append("id", id);

    const res = await fetch('/upload_drive_only', { method: 'POST', body: formData });
    const data = await res.json();
    document.getElementById('result').innerText = JSON.stringify(data, null, 2);
    if (data.link) window.open(data.link, "_blank");
}
</script>
</body>
</html>
"""

# ======================================================
# PREDICT ROUTE
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    img_file = request.files["image"]
    local_path = os.path.join(UPLOAD_FOLDER, secure_filename(img_file.filename))
    img_file.save(local_path)

    try:
        img = image.load_img(local_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, 0)
        x = preprocess_input(x)
        if is_keras_model:
            preds = model.predict(x)
        else:
            preds = list(infer(tf.constant(x, dtype=tf.float32)).values())[0].numpy()
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        class_name = index_to_class[class_id] if index_to_class else str(class_id)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

    return jsonify({"predicted_class": class_name, "confidence": confidence})

# ======================================================
# UPLOAD TO DRIVE (WITH SAFE CUSTOMER INFO)
# ======================================================
@app.route("/upload_drive_only", methods=["POST"])
def upload_drive_only():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    cust_name = request.form.get("name", "").strip()
    cust_id = request.form.get("id", "").strip()

    if not cust_name or not cust_id:
        return jsonify({"error": "Missing customer name or ID"}), 400

    # Safe filename
    safe_name = cust_name.replace(" ", "_").replace("/", "_")
    safe_id = cust_id.replace(" ", "_").replace("/", "_")

    local_path = os.path.join(UPLOAD_FOLDER, secure_filename(img_file.filename))
    img_file.save(local_path)

    base_name, ext = os.path.splitext(img_file.filename)
    filename_with_customer = f"{base_name}_{safe_name}_{safe_id}{ext}"

    try:
        result = upload_to_drive(local_path, filename_with_customer)
        result["customer_name"] = cust_name
        result["customer_id"] = cust_id
        result["uploaded_filename"] = filename_with_customer
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

    return jsonify(result)

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
