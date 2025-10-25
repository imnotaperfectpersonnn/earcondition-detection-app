# -*- coding: utf-8 -*-
"""Automated Ear Disease Detection Streamlit App — YOLOv12 (Dark Edition)"""

import streamlit as st
from PIL import Image
import tempfile
import os
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Automated Ear Disease Detection", layout="wide")

# --- Dark Neon Style ---
dark_neon_style = """
<style>
.stApp {
  background: linear-gradient(180deg, #000000, #0a0a0a);
  color: #e0e0e0;
  font-family: 'Segoe UI', sans-serif;
}
.stApp .main {
  background-color: rgba(20, 20, 20, 0.95);
  backdrop-filter: blur(12px);
  margin: 2rem auto;
  padding: 2rem 3rem;
  border-radius: 20px;
  box-shadow: 0 0 20px rgba(0,255,255,0.05);
  max-width: 1100px;
}
h1, h2, h3 {
  color: #00ffff;
  text-shadow: 0 0 8px rgba(0,255,255,0.5);
}
.stButton>button {
  background: linear-gradient(90deg, #00ffff, #0077ff);
  color: #000;
  border-radius: 10px;
  padding: 0.6rem 1.2rem;
  border: none;
  font-weight: 600;
  box-shadow: 0 0 15px rgba(0,255,255,0.3);
  transition: 0.3s ease;
}
.stButton>button:hover {
  background: linear-gradient(90deg, #00bfff, #00ffff);
  transform: translateY(-2px);
  box-shadow: 0 0 25px rgba(0,255,255,0.6);
}
.stSlider, .stFileUploader, .stInfo, .stCaption {
  color: #e0e0e0 !important;
}
</style>
"""
st.markdown(dark_neon_style, unsafe_allow_html=True)

# --- Header ---
st.title("Automated Ear Disease Detection")
st.caption("Detect ear conditions using YOLOv12 object detection model")

col1, col2 = st.columns([1, 2])

# --- Inputs ---
with col1:
    st.header("Inputs")
    uploaded_image = st.file_uploader("Upload an otoscopic image", type=["png", "jpg", "jpeg"])
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    run = st.button("Run Inference")

# --- Preview / Results ---
with col2:
    st.header("Preview / Result")
    if uploaded_image is None:
        st.info("Upload an otoscopic image to begin analysis.")
    else:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Input image", use_container_width=True)
        if not run:
            st.caption("Click **Run Inference** to detect possible ear conditions.")

# --- Model Loader ---
def load_yolov12_model(path):
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Could not load YOLOv12 model: {e}")
        return None

# --- Dummy Fallback ---
def dummy_inference_pil(image_pil):
    import PIL.ImageDraw as ImageDraw
    im = image_pil.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    box = (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))
    draw.rectangle(box, outline="#00ffff", width=5)
    draw.text((box[0], box[1]-25), "Detected: Possible Condition (0.99)", fill="#00ffff")
    return im

# --- Main Logic ---
if run:
    if uploaded_image is None:
        st.error("Please upload an image first.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_image.getbuffer())
        tfile.flush()
        tfile.close()

        model_path = "best.pt"
        result_image = None

        if os.path.exists(model_path):
            model = load_yolov12_model(model_path)
            if model is not None:
                st.info("Running inference using YOLOv12 model...")
                try:
                    results = model(source=tfile.name, conf=conf, verbose=False)
                    r = results[0]

                    # Use plot() to get annotated image array (no GUI needed)
                    annotated = r.plot()
                    result_image = Image.fromarray(annotated)

                except Exception as exc:
                    st.error(f"Model inference failed: {exc}")
                    # Fallback dummy image
                    result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))
            else:
                st.warning("Model failed to load. Showing dummy result.")
                result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))
        else:
            st.warning("Model file (best.pt) not found. Showing dummy result.")
            result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))

        if result_image is not None:
            st.image(result_image, caption="Detection Result", use_container_width=True)
