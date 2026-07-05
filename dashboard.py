import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import plotly.express as px
import re

# ==========================
# CONFIGURATION
# ==========================
PREDICTIONS_DIR = "predictions"
IMAGES_DIR = "test_images" 

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
# METADATA PARSING (Customer Info Extraction)
# ==========================================
def extract_customer_info(filename):
    match = re.search(r'_(?P<name>[a-zA-Z]+)_(?P<id>\d+)\.', filename)
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
st.info(
    f"📦 **Latest Item:** `{latest_row['predicted_class_name']}` | "
    f"👤 **Customer:** `{latest_row['Customer Name']}` (ID: {latest_row['Customer ID']}) | "
    f"🎯 **Confidence:** `{latest_row['confidence'] * 100:.2f}%` | "
    f"⏱️ **Timestamp:** `{latest_row['timestamp']}`"
)

# ==========================================
# SIDEBAR FILTER (Customer Segmentation)
# ==========================================
st.sidebar.header("🔍 Filter Options")
all_customers = sorted(df['Customer Name'].unique())
selected_customer = st.sidebar.selectbox("Filter by Customer Name:", ["All Customers"] + all_customers)

if selected_customer != "All Customers":
    display_df = df[df['Customer Name'] == selected_customer]
else:
    display_df = df

# ==========================
# DISPLAY DATAFRAME
# ==========================
st.subheader(f"📊 Predictions Table ({selected_customer})")
st.dataframe(
    display_df[['filename', 'Customer Name', 'Customer ID', 'true_label', 'predicted_class_name', 'is_correct', 'confidence', 'timestamp']], 
    use_container_width=True
)

# ==========================
# IMAGE PREVIEW
# ==========================
st.subheader("🖼️ Image Preview")

if os.path.exists(IMAGES_DIR):
    selected_image = st.selectbox("Choose an image to preview:", display_df["filename"].tolist())
    img_path = os.path.join(IMAGES_DIR, selected_image)

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
        st.warning(f"Image not found locally: {img_path}")
else:
    st.warning("Image folder not found. Upload test_images to view previews.")

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