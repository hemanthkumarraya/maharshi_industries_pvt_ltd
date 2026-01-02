import streamlit as st
import os

OUTPUT_DIR = "Output"

st.set_page_config(page_title="Output Viewer", layout="wide")
st.title("📁 Output Viewer")

if not os.path.exists(OUTPUT_DIR):
    st.error("❌ Output folder not found")
    st.stop()

files = sorted(os.listdir(OUTPUT_DIR))

if not files:
    st.warning("⚠️ Output folder is empty")

for file in files:
    file_path = os.path.join(OUTPUT_DIR, file)

    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        st.image(file_path, caption=file, use_column_width=True)

    elif file.lower().endswith((".mp4", ".avi", ".mov")):
        st.video(file_path)

    else:
        st.text(f"Unsupported file: {file}")
