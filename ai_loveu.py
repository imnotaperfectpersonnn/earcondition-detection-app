# -*- coding: utf-8 -*-
"""Ai-love-u.ipynb
Final Streamlit app — YOLOv12 (Ultralytics 8.3+)
"""

import streamlit as st
from PIL import Image
import tempfile
import os
import numpy as np

# --- Styling ---
blue_overlay = """
<style>
.stApp {
  background: linear-gradient(120deg, #89f7fe, #66a6ff);
  min-height: 100vh;
}
.stApp .main {
  background-color: rgba(255, 255, 255, 0.65);
  backdrop-filter: blur(8px);
  margin: 2rem;
  padding: 2rem;
  border-radius: 20px;
  box-shadow: 0 4px 25px rgba(0,0,0,0.15);
}
</style>
"""

st.markdown(blue_overlay, unsafe_allow_html=True)
st.set_page_config(page_title="Automated Ear Disease Detection", layout="wide")

st.title("Automated Ear Disease Detection through Object Detection of Otoscopic Images")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Inputs")
    uploaded_image = st.file_uploader("Upload an otoscopic image", type=["png", "jpg", "jpeg"])
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    run = st.button("Run inference")

with col2:
    st.header("Preview / Result")
    if uploaded_image is None:
        st.info("Upload an otoscopic image to begin analysis.")
    else:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Input image", use_column_width=True)
        if not run:
            st.caption("Click **Run inference** to detect possible ear conditions.")

# --- Model Loader ---
def load_yolov12_model(path):
    try:
        from ultralytics import YOLO  # YOLOv12-compatible
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Could not load YOLOv12 model: {e}")
        return None

# --- Dummy Fallback ---
def dummy_inference_pil(image_pil):
    import PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
    im = image_pil.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    box = (int(w*0.15), int(h*0.15), int(w*0.75), int(h*0.75))
    draw.rectangle(box, outline="red", width=6)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
    except:
        font = ImageFont.load_default()
    draw.text((box[0], box[1]-24), "possible_condition:0.99", fill="red", font=font)
    return im

# --- Main Logic ---
if run:
    if uploaded_image is None:
        st.error("Please upload an image first.")
    else:
        # Save uploaded image
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_image.getbuffer())
        tfile.flush()
        tfile.close()

        # Path to Google Drive model
        gdrive_model_path = "/content/drive/MyDrive/Colab Notebooks/Ear_Dataset/logs_for_visualization/train/weights/best.pt"

        model = None
        if os.path.exists(gdrive_model_path):
            model = load_yolov12_model(gdrive_model_path)
        else:
            st.warning("Model file not found in Google Drive path. Running dummy demo.")
        
        result_image = None

        if model is not None:
            st.info("Running inference using YOLOv12 model...")
            try:
                results = model(source=tfile.name, conf=conf, verbose=False)
                r = results[0]
                annotated = r.plot()
                annotated_pil = Image.fromarray(annotated)
                result_image = annotated_pil
            except Exception as exc:
                st.error(f"Model inference failed: {exc}")
                result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))
        else:
            result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))

        if result_image is not None:
            st.image(result_image, caption="Detection Result", use_column_width=True)
