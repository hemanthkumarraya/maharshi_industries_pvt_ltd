import streamlit as st
import requests
import os

# ================= CONFIG =================
GITHUB_USER = "hemanthkumarraya"
REPO_NAME = "maharshi_industries_pvt_ltd"
BRANCH = "main"
OUTPUT_FOLDER = "Output"

LOCAL_CACHE = "cache_output"
os.makedirs(LOCAL_CACHE, exist_ok=True)

# ==========================================

st.set_page_config(page_title="Output Viewer", layout="wide")
st.title("📁 Output Viewer (Loaded from GitHub)")

@st.cache_data(show_spinner=True)
def fetch_file_list():
    """Get file list from GitHub Output folder"""
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/contents/{OUTPUT_FOLDER}?ref={BRANCH}"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

@st.cache_data(show_spinner=True)
def download_file(file_name, download_url):
    """Download file once and cache locally"""
    local_path = os.path.join(LOCAL_CACHE, file_name)
    if not os.path.exists(local_path):
        r = requests.get(download_url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

try:
    files = fetch_file_list()
except Exception as e:
    st.error("❌ Failed to fetch files from GitHub")
    st.stop()

if not files:
    st.warning("⚠️ No files found in Output folder")

for file in files:
    if file["type"] != "file":
        continue

    file_name = file["name"]
    download_url = file["download_url"]

    local_file = download_file(file_name, download_url)

    col1, col2 = st.columns([5, 1])

    with col1:
        if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            st.image(local_file, caption=file_name, use_column_width=True)

        elif file_name.lower().endswith((".mp4", ".avi", ".mov")):
            st.video(local_file)

        else:
            st.text(file_name)

    with col2:
        with open(local_file, "rb") as f:
            st.download_button(
                "⬇️",
                data=f,
                file_name=file_name,
                mime="application/octet-stream",
                key=file_name
            )

    st.divider()
