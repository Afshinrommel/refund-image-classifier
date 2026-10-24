# Automated Refund Item Classification Pipeline

An end-to-end, decoupled MLOps system that automates the categorization of returned e-commerce merchandise using deep learning (MobileNetV2), a Zero-SQL metadata tracking architecture, scheduled batch processing with GitHub Actions, and an interactive Streamlit monitoring dashboard.

---

## 📌 Project Overview & Architecture

High return volumes in growing e-commerce businesses create operational bottlenecks and high costs when handled manually. This project replaces manual inspection with an automated, four-phase machine learning pipeline:

1. **Phase 1: Offline Model Training (`image_refund_code.ipynb`)**
   Downloads the Caltech-101 dataset via Kaggle API, performs data augmentation and transfer learning using MobileNetV2, fine-tunes deep convolutional layers, and exports deployment artifacts (`final_model_caltech.h5` and `class_indices.json`).

2. **Phase 2: Client Web Interface & Ingestion (`api_server_two_button_v4.py`)**
   A Flask web application running locally. It validates customer credentials against `customers.json`, runs real-time inference in memory (RAM), authenticates user access with Google OAuth 2.0 (`client_secret.json`), and uploads return photos to a Google Drive staging folder using a standardized filename schema.

3. **Phase 3: Automated Nightly Batch Processing (`main.py` & GitHub Actions)**
   An automated MLOps workflow (`.github/workflows/main.yml`) scheduled at 02:00 UTC. It downloads staging images and model artifacts from Google Drive using `gdown`, filters out previously processed files to prevent redundant compute, predicts categories for new returns, updates the central `predictions/predictions.csv` log, and generates visual accuracy metrics.

4. **Phase 4: Monitoring & Analytics Dashboard (`dashboard.py`)**
   A live business intelligence interface built with Streamlit and Plotly. It parses customer metadata and timestamps directly from filenames, displays chronological predictions, allows single-customer filtering, generates category volume charts, and dynamically fetches original images from Google Drive for audit inspection.

---

## 📂 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── main.yml               # Scheduled GitHub Actions cronjob for nightly batch processing
├── predictions/
│   ├── predictions.csv            # Central audit log storing historical batch predictions
│   └── accuracy_chart.png         # Dynamically generated model performance pie chart
├── api_server_two_button_v4.py    # Flask REST API and client-facing web application
├── main.py                        # Batch inference worker executed in GitHub Actions runner
├── dashboard.py                   # Streamlit operational monitoring and analytics dashboard
├── image_refund_code.ipynb        # Model training and fine-tuning pipeline (Google Colab / Jupyter)
├── customers.json                 # Authorized customer database for identity validation
├── class_indices.json             # Numeric index-to-class label dictionary
├── client_secret.json             # Google Cloud OAuth 2.0 Web Client credentials
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation and reproduction guide
```

---

## ⚙️ Prerequisites & Configuration Files

Before running the application locally or in production, verify the presence of the following configuration files in the root directory:

* **`client_secret.json`:** OAuth 2.0 client secrets file obtained from Google Cloud Console (APIs & Services > Credentials). Required by `api_server_two_button_v4.py` for customer authentication and Drive upload permissions.
* **`customers.json`:** JSON array containing authorized user records (`{"name": "...", "id": "..."}`) used by `/validate_customer`.
* **`class_indices.json`:** JSON mapping generated during training to decode numeric predictions into product category names.
* **`final_model_caltech.h5`:** Pre-trained Keras model weights. If not placed locally, `main.py` downloads it automatically from Google Drive via `gdown`.
* **Environment Variables:**
  * `OAUTHLIB_INSECURE_TRANSPORT=1`: Required for local development over HTTP (`http://localhost:5000`).
  * `TF_USE_LEGACY_KERAS=1`: Ensures Keras 2 compatibility with TensorFlow 2.19.

*(Note: `service_account.json` is not required; batch downloads and dashboard previews use public/shared Drive links via `gdown`, while client uploads use interactive OAuth 2.0 user credentials).*

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Afshinrommel/refund-image-classifier.git](https://github.com/Afshinrommel/refund-image-classifier.git)
cd refund-image-classifier
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🛠️ Step-by-Step Execution Guide

### Step 1: Model Training (Optional / Offline)
To re-train the classifier or generate fresh weights:
* Open `image_refund_code.ipynb` in Google Colab or a local Jupyter environment.
* Ensure your Kaggle API key (`kaggle.json`) is configured to pull the Caltech-101 dataset.
* Run all cells to fine-tune MobileNetV2 and export `final_model_caltech.h5` and `class_indices.json`.

### Step 2: Running the Client-Facing Flask Server
1. Start the Flask service:
   ```bash
   export OAUTHLIB_INSECURE_TRANSPORT=1   # Linux/macOS
   # set OAUTHLIB_INSECURE_TRANSPORT=1    # Windows CMD
   python api_server_two_button_v4.py
   ```
2. Navigate to `http://localhost:5000/upload` in your browser.
3. Click **"🔑 Authorize Google Drive"** to sign in with your Google test account.
4. Enter an authorized customer name and ID (e.g., `Afshin` / `22222`) and click **"✅ Validate Customer"**.
5. Select a product image:
   * Click **"🔍 Predict"** for instant RAM inference.
   * Click **"☁️ Upload to Drive"** to stage the item for nightly processing.

### Step 3: Triggering Batch Inference (Local Simulation)
To run the automated batch job locally without waiting for GitHub Actions:
```bash
python main.py
```
The script will pull pending images from the staging folder, skip previously processed records, perform batch inference, append new records to `predictions/predictions.csv`, and update `predictions/accuracy_chart.png`.

### Step 4: Launching the Monitoring Dashboard
Run the Streamlit analytics interface:
```bash
streamlit run dashboard.py
```
* **KPI Card:** View details of the most recently processed return.
* **Customer Filter:** Filter data by customer in the sidebar to inspect individual return habits.
* **Auditing:** Click any row in the prediction table to dynamically download and view the original item photo.
* **Summary Charts:** Inspect the distribution of returned product categories.

---

## 🏗️ Architectural Decisions & Constraints

* **Zero-SQL Metadata Persistence:** Customer IDs, names, and upload timestamps are directly encoded into uploaded filenames (e.g., `watch_Afshin_22222_20260709-182338.png`). Metadata is parsed using Regular Expressions, avoiding the overhead, licensing, and schema migrations of a traditional relational database.
* **Staging Strategy:** Google Drive serves as a low-cost, serverless staging data lake for the MVP, replacing complex cloud object storage (such as AWS S3).
* **Inference Decoupling:** Heavy deep learning classification is separated from the web server into a scheduled, off-peak batch runner (GitHub Actions), avoiding CPU starvation and latency during peak customer hours.