# -*- coding: utf-8 -*-
"""Automated Ear Disease Detection Streamlit App — YOLOv12 (Ultralytics 8.3+)"""

import streamlit as st
from PIL import Image
import tempfile
import os

# --- Page Config ---
st.set_page_config(page_title="Automated Ear Disease Detection", layout="wide")

# --- Modern Dark Style ---
modern_dark_style = """
<style>
.stApp {
  background: linear-gradient(135deg, #0e1117, #1b1f29);
  color: #f1f1f1;
  font-family: 'Segoe UI', sans-serif;
}
.stApp .main {
  background-color: rgba(25, 28, 36, 0.85);
  backdrop-filter: blur(12px);
  margin: 2rem auto;
  padding: 2rem 3rem;
  border-radius: 20px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.4);
  max-width: 1100px;
}
h1, h2, h3 {
  color: #82b1ff;
}
.stButton>button {
  background: linear-gradient(90deg, #004aad, #0078ff);
  color: white;
  border-radius: 10px;
  padding: 0.6rem 1.2rem;
  border: none;
  font-weight: 600;
  box-shadow: 0 4px 15px rgba(0,0,0,0.3);
  transition: 0.3s ease;
}
.stButton>button:hover {
  background: linear-gradient(90deg, #0078ff, #00c6ff);
  transform: translateY(-2px);
}
.stSlider, .stFileUploader, .stInfo, .stCaption {
  color: #f1f1f1 !important;
}
</style>
"""
st.markdown(modern_dark_style, unsafe_allow_html=True)

# --- App Header ---
st.title("Automated Ear Disease Detection")
st.caption("Detect ear conditions using YOLOv12 object detection model")

col1, col2 = st.columns([1, 2])

# --- Input Section ---
with col1:
    st.header("Inputs")
    uploaded_image = st.file_uploader("Upload an otoscopic image", type=["png", "jpg", "jpeg"])
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    run = st.button("Run Inference")

# --- Preview Section ---
with col2:
    st.header("Preview / Result")
    if uploaded_image is None:
        st.info("Upload an otoscopic image to begin analysis.")
    else:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Input image", use_container_width=True)
        if not run:
            st.caption("Click **Run Inference** to detect possible ear conditions.")

# --- Dummy Fallback (if model fails) ---
def dummy_inference_pil(image_pil):
    import PIL.ImageDraw as ImageDraw
    im = image_pil.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    box = (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))
    draw.rectangle(box, outline="#00c6ff", width=5)
    draw.text((box[0], box[1]-25), "Detected: Possible Condition (0.99)", fill="#00c6ff")
    return im

# --- Safe Model Loader ---
def load_yolov12_model(path):
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        return model
    except Exception as e:
        st.warning(f"YOLO model could not be loaded: {e}")
        return None

# --- Safe Inference Function ---
def run_inference(model, image_path, conf):
    try:
        results = model(source=image_path, conf=conf, verbose=False)
        r = results[0]
        annotated = r.plot()  # may fail if libGL missing
        return Image.fromarray(annotated)
    except Exception as e:
        st.warning(f"Model inference failed: {e}. Using dummy output instead.")
        return dummy_inference_pil(Image.open(image_path).convert("RGB"))

# --- Main Logic ---
if run:
    if uploaded_image is None:
        st.error("Please upload an image first.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_image.getbuffer())
        tfile.flush()
        tfile.close()

        model_path = "best.pt"  # make sure this exists in repo root
        result_image = None

        if os.path.exists(model_path):
            model = load_yolov12_model(model_path)
            if model is not None:
                st.info("Running inference using YOLOv12 model...")
                result_image = run_inference(model, tfile.name, conf)
            else:
                st.warning("Model failed to load. Showing dummy result.")
                result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))
        else:
            st.warning("Model file (best.pt) not found in repository. Showing dummy result.")
            result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))

        if result_image is not None:
            st.image(result_image, caption="Detection Result", use_container_width=True)
