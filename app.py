import os
import gdown
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import EnhancedResNet50GAT

MODEL_PATH = "enhanced_model.pth"

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection & Classification")


@st.cache_resource
def load_tumor_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1mhW8fp31-sb3Bv-UYuXTg8bQrARx_6ki"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Initialize model with 4 classes
    model = EnhancedResNet50GAT(pretrained=False, num_classes=4)

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_tumor_model()

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

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
    st.image(image, caption="Uploaded MRI Scan", width=300)

    img_tensor = transform(image).unsqueeze(0)

    if st.button("Classify Scan"):
        with torch.no_grad():
            log_probs = model(img_tensor)
            # Convert log_softmax outputs to probabilities using exp()
            probs = torch.exp(log_probs[0]).numpy()

        st.write("### Prediction Results")
        for label, prob in zip(CLASSES, probs):
            st.progress(float(prob), text=f"{label}: {prob * 100:.1f}%")
