import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image

# ==========================
# ⚙️ CONFIGURATION
# ==========================
PREDICTIONS_DIR = "predictions"
IMAGES_DIR = "test_images"  # folder containing input images

st.set_page_config(
    page_title="Refund Item Classifier Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧾 Refund Item Classification Dashboard")
st.caption("Automatically generated predictions from the nightly/batch process")

# ==========================
# 🗂️ LOAD AVAILABLE CSV FILES
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
# 📅 SELECT WHICH RUN TO VIEW
# ==========================
selected_csv = st.selectbox("Select prediction run:", csv_files)
csv_path = os.path.join(PREDICTIONS_DIR, selected_csv)

df = pd.read_csv(csv_path)
st.success(f"Loaded {len(df)} predictions from: `{selected_csv}`")

# ==========================
# 📊 DISPLAY DATAFRAME
# ==========================
st.subheader("📊 Predictions Table")
st.dataframe(df)

# ==========================
# 🖼️ IMAGE PREVIEW
# ==========================
st.subheader("🖼️ Image Preview")

if os.path.exists(IMAGES_DIR):
    selected_image = st.selectbox("Choose an image to preview:", df["filename"].tolist())
    img_path = os.path.join(IMAGES_DIR, selected_image)

    if os.path.exists(img_path):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(Image.open(img_path), caption=selected_image, use_container_width=True)
        with col2:
            row = df[df["filename"] == selected_image].iloc[0]
            st.markdown(f"**Predicted Class:** {row['predicted_class_name']}")
            st.markdown(f"**Confidence:** {row['confidence'] * 100:.2f}%")
    else:
        st.warning(f"Image not found locally: {img_path}")
else:
    st.warning("Image folder not found. Upload test_images to view previews.")

# ==========================
# 🧩 SUMMARY STATISTICS
# ==========================
st.subheader("📈 Summary Statistics")

summary = (
    df.groupby("predicted_class_name")
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

st.bar_chart(summary.set_index("predicted_class_name"))

# ==========================
# 📅 FOOTER
# ==========================
st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Streamlit Dashboard by Refund AI System")
