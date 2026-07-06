import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import plotly.express as px
import re
import gdown

# ==========================
# CONFIGURATION
# ==========================
PREDICTIONS_DIR = "predictions"
IMAGES_DIR = "test_images" 
# Direct link to the customer test images folder in Google Drive
TEST_FOLDER_URL = "https://drive.google.com/drive/folders/1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc?usp=drive_link"

st.set_page_config(
    page_title="Refund Item Classifier Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧾 Refund Item Classification Dashboard")
st.caption("Automatically generated predictions from the nightly/batch process")

# ==========================
# LOAD AVAILABLE CSV FILES
# ==========================
if not os.path.exists(PREDICTIONS_DIR):
    st.warning("No predictions folder found yet.")
    st.stop()

csv_files = sorted(
    [f for f in os.listdir(PREDICTIONS_DIR) if f.endswith(".csv")],
    reverse=True
)

if not csv_files:
    st.warning("No prediction CSVs available yet.")
    st.stop()

# ==========================
# SELECT WHICH RUN TO VIEW
# ==========================
selected_csv = st.selectbox("Select prediction run:", csv_files)
csv_path = os.path.join(PREDICTIONS_DIR, selected_csv)

df = pd.read_csv(csv_path)

# ==========================================
# FULLY AUTOMATED TIMESTAMP EXTRACTION
# ==========================================
def extract_upload_time(filename):
    # Automatically extract timestamp from the new filename pattern using regex
    match = re.search(r'_(?P<time>\d{8}-\d{6})\.', filename)
    if match:
        try:
            return datetime.strptime(match.group('time'), "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
            
    # Automated fallback map for legacy images uploaded before the pipeline update
    legacy_map = {
        "laptop_5_Afshin_22222.png": "2026-04-14",
        "watch_8_Afshin_22222.png": "2026-04-21",
        "guitar_1_Amir_11111.png": "2026-04-14",
        "laptop_Alice_12345.png": "2025-11-08",
        "image_0001_Amir_11111.jpg": "2026-07-06",
        "watch_5_Afshin_22222.png": "2026-04-14",
        "laptop_7_hooshang_989898.png": "2026-07-05",
        "watch_Bob_67890.png": "2025-11-10",
        "moto_Alice_12345.jpg": "2025-11-10",
        "watch_1_Amir_11111.png": "2026-04-14",
        "laptop_1_Amir_11111.png": "2026-04-14",
        "watch_0_Ali_00000.png": "2026-04-14",
        "watch_91_hooshang_989898.png": "2026-07-05",
        "guitar_2_Afshin_22222.png": "2026-04-14",
        "guitar_0_Ali_00000.png": "2026-04-14",
        "watch_4_Afshin_22222.png": "2026-04-14",
        "laptop_0_Ali_00000.png": "2026-04-14",
        "watch_6_Afshin_22222.png": "2026-04-14",
        "watch_92_hooshang_989898.png": "2026-07-05",
        "watch_9_Afshin_22222.png": "2026-04-22",
        "laptop_6_hooshang_989898.png": "2026-07-05",
        "guitar_Alice_12345.png": "2025-11-08",
        "laptop_Charlie_24680.png": "2025-11-08"
    }
    return legacy_map.get(filename, "N/A")

# Apply automated timestamps to the dataframe
df['gdrive_upload_time'] = df['filename'].apply(extract_upload_time)

# ==========================================
# METADATA PARSING (Customer Info Extraction)
# ==========================================
def extract_customer_info(filename):
    # Updated regex to handle filenames both with and without automated timestamps
    match = re.search(r'_(?P<name>[a-zA-Z]+)_(?P<id>\d+)', filename)
    if match:
        return match.group('name'), match.group('id')
    return "Unknown", "Unknown"

parsed_info = df['filename'].apply(extract_customer_info)
df['Customer Name'] = [info[0] for info in parsed_info]
df['Customer ID'] = [info[1] for info in parsed_info]

st.success(f"Loaded {len(df)} predictions from: `{selected_csv}`")

# ==========================================
# LATEST REFUND (KPI Summary Card)
# ==========================================
st.markdown("---")
st.subheader("🔔 Latest Processed Refund")
latest_row = df.iloc[-1]
upload_time = latest_row.get('gdrive_upload_time', 'N/A')

st.info(
    f"📦 **Latest Item:** `{latest_row['predicted_class_name']}` | "
    f"👤 **Customer:** `{latest_row['Customer Name']}` (ID: {latest_row['Customer ID']}) | "
    f"🎯 **Confidence:** `{latest_row['confidence'] * 100:.2f}%` | "
    f"⏱️ **Upload Time (Drive):** `{upload_time}`"
)

# ==========================================
# SIDEBAR FILTER (Customer Segmentation)
# ==========================================
st.sidebar.header("🔍 Filter Options")
all_customers = sorted(df['Customer Name'].unique())
selected_customer = st.sidebar.selectbox("Filter by Customer Name:", ["All Customers"] + all_customers)

if selected_customer != "All Customers":
    display_df = df[df['Customer Name'] == selected_customer].copy()
else:
    display_df = df.copy()

# ==========================
# DISPLAY DATAFRAME
# ==========================
st.subheader(f"📊 Predictions Table ({selected_customer})")

columns_to_display = [
    'filename', 'Customer Name', 'Customer ID', 'true_label', 
    'predicted_class_name', 'is_correct', 'confidence', 
    'gdrive_upload_time', 'timestamp'
]

st.dataframe(
    display_df[columns_to_display], 
    use_container_width=True
)

# ==========================
# IMAGE PREVIEW & GOOGLE DRIVE SYNC
# ==========================
st.subheader("🖼️ Image Preview")

selected_image = st.selectbox("Choose an image to preview:", display_df["filename"].tolist())
img_path = os.path.join(IMAGES_DIR, selected_image)

if not os.path.exists(img_path):
    with st.spinner("Downloading customer images from Google Drive... Please wait."):
        try:
            gdown.download_folder(url=TEST_FOLDER_URL, output=IMAGES_DIR, quiet=True, use_cookies=False)
        except Exception as e:
            st.error(f"Failed to download images from Google Drive: {str(e)}")

if os.path.exists(img_path):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(Image.open(img_path), caption=selected_image, use_container_width=True)
    with col2:
        row = df[df["filename"] == selected_image].iloc[0]
        st.markdown(f"**Customer:** {row['Customer Name']} (ID: {row['Customer ID']})")
        st.markdown(f"**Predicted Class:** {row['predicted_class_name']}")
        st.markdown(f"**Confidence:** {row['confidence'] * 100:.2f}%")
else:
    st.warning("Image could not be loaded. Please ensure the Google Drive folder contains this file.")

# ==========================
# SUMMARY STATISTICS
# ==========================
st.subheader(f"📈 Summary Statistics ({selected_customer})")

summary = (
    display_df.groupby("predicted_class_name")
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

fig = px.bar(
    summary,
    x='predicted_class_name',
    y='count',
    text='count',
    color_discrete_sequence=['#FF4B2B'], 
    labels={'predicted_class_name': 'Category', 'count': 'Quantity'}
)

fig.update_layout(
    font=dict(size=22, color="white"), 
    xaxis=dict(tickfont=dict(size=20)), 
    yaxis=dict(tickfont=dict(size=20)), 
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=600
)

fig.update_traces(
    textfont_size=25,   
    textposition='outside',
    marker_line_color='white',
    marker_line_width=2
)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Streamlit Dashboard by Refund AI System")