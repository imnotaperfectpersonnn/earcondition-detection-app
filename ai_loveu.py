# -*- coding: utf-8 -*-
"""Automated Ear Disease Detection Streamlit App — YOLOv12 (Ultralytics 8.3+)"""

import streamlit as st
from PIL import Image
import tempfile
import os
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Automated Ear Disease Detection", layout="wide")

# --- Styling ---
modern_style = """
<style>
.stApp {
  background: linear-gradient(135deg, #f0f2f5, #dfe6ee);
  font-family: 'Segoe UI', sans-serif;
}
.stApp .main {
  background-color: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  margin: 2rem auto;
  padding: 2rem 3rem;
  border-radius: 20px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.1);
  max-width: 1100px;
}
h1, h2, h3 {
  color: #003366;
}
.stButton>button {
  background: #0052cc;
  color: white;
  border-radius: 8px;
  padding: 0.5rem 1rem;
  border: none;
  font-weight: bold;
}
.stButton>button:hover {
  background: #003d99;
}
</style>
"""
st.markdown(modern_style, unsafe_allow_html=True)

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

# --- Model Loader ---
def load_yolov12_model(path):
    try:
        from ultralytics import YOLO  # YOLOv12-compatible
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Could not load YOLOv12 model: {e}")
        return None

# --- Dummy Fallback (if model fails) ---
def dummy_inference_pil(image_pil):
    import PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
    im = image_pil.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    box = (int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))
    draw.rectangle(box, outline="red", width=6)
    draw.text((box[0], box[1]-25), "Detected: Possible Condition (0.99)", fill="red")
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

        model_path = "best.pt"  # now points directly to the file in your repo
        result_image = None

        if os.path.exists(model_path):
            model = load_yolov12_model(model_path)
            if model is not None:
                st.info("Running inference using YOLOv12 model...")
                try:
                    results = model(source=tfile.name, conf=conf, verbose=False)
                    r = results[0]
                    annotated = r.plot()
                    result_image = Image.fromarray(annotated)
                except Exception as exc:
                    st.error(f"Model inference failed: {exc}")
                    result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))
            else:
                st.warning("Model failed to load. Showing dummy result.")
                result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))
        else:
            st.warning("Model file (best.pt) not found in repository. Showing dummy result.")
            result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))

        if result_image is not None:
            st.image(result_image, caption="Detection Result", use_container_width=True)
