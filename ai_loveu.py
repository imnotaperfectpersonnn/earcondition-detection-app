import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import os

# ============================
# Modern Dark UI Styling
# ============================
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #e0e0e0;
        }
        [data-testid="stSidebar"] {
            background-color: #1c1f26;
        }
        h1, h2, h3 {
            color: #f5f5f5;
        }
        button {
            background-color: #31363f !important;
            color: white !important;
            border-radius: 8px !important;
        }
        button:hover {
            background-color: #444b57 !important;
        }
        .stFileUploader {
            background-color: #1a1d23;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 10px;
        }
        .css-1cpxqw2 {
            background-color: #1e2229 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================
# Title
# ============================
st.title("🧠 AI Love U - Ear Disease Detection")
st.write("Upload an ear image below to detect conditions using your YOLOv12 model.")

# ============================
# Load Model
# ============================
model_path = "best.pt"

if not os.path.exists(model_path):
    st.error("Model file 'best.pt' not found! Please upload it to the same folder as this app.")
    st.stop()

model = YOLO(model_path)
st.success("Model loaded successfully ✅")

# ============================
# Image Upload
# ============================
uploaded_image = st.file_uploader("Upload an ear image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting... Please wait"):
            results = model(image)
            annotated_frame = results[0].plot(line_width=2, labels=True, boxes=True)

            # Convert result to PIL Image
            result_image = Image.fromarray(annotated_frame)
            st.image(result_image, caption="Detection Result", use_container_width=True)

            # Print detected classes and confidence
            detections = []
            for box in results[0].boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls] if hasattr(model, 'names') else f"class {cls}"
                detections.append((label, f"{conf:.2f}"))

            if detections:
                st.subheader("Detections:")
                for label, conf in detections:
                    st.write(f"• **{label}** — Confidence: {conf}")
            else:
                st.warning("No detections found.")

else:
    st.info("Please upload an image to start detection.")
