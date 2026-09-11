import os
import gdown
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

MODEL_PATH = "enhanced_model.pth"

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor Detection & Classification")


@st.cache_resource
def load_tumor_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1mhW8fp31-sb3Bv-UYuXTg8bQrARx_6ki"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Load model with PyTorch onto CPU
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
    return model


# Load PyTorch model once
model = load_tumor_model()

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Image Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

uploaded_file = st.file_uploader(
    "Upload MRI Scan", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", width=300)

    # Convert image to tensor and add batch dimension
    img_tensor = transform(image).unsqueeze(0)

    if st.button("Classify Scan"):
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0).numpy()

        st.write("### Prediction Results")
        for label, prob in zip(CLASSES, probs):
            st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
