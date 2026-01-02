import streamlit as st
import os

OUTPUT_DIR = "Output"

st.title("Output Viewer")

files = os.listdir(OUTPUT_DIR)

for file in files:
    path = os.path.join(OUTPUT_DIR, file)

    if file.endswith((".jpg", ".png", ".jpeg")):
        st.image(path, caption=file)

    elif file.endswith((".mp4", ".avi", ".mov")):
        st.video(path)
