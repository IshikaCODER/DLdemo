import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic

# Import model components from model.py
from model import EnhancedResNet50GAT, _unnorm

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Classifier (ResNet-GAT)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Class Labels mapping ---
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_DESCRIPTIONS = {
    "Glioma": "Tumor arising from glial cells in the brain or spine.",
    "Meningioma": "Tumor arising from the meninges surrounding the brain.",
    "No Tumor": "No prominent tumor structures detected in the provided slice.",
    "Pituitary": "Tumor occurring in the pituitary gland at the base of the brain."
}

# --- Cache Model Loading ---
@st.cache_resource
def load_trained_model(model_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedResNet50GAT(pretrained=False, num_classes=4)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- Image Preprocessing ---
def transform_image(image: Image.Image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

# --- Superpixel Overlay Visualizer ---
def plot_superpixel_overlay(img_tensor):
    img_np = _unnorm(img_tensor.squeeze(0))
    segs = slic(img_np.astype(np.float64), n_segments=50, compactness=10, start_label=0, channel_axis=2)
    overlay = mark_boundaries(img_np, segs, color=(1, 0, 0))
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img_np)
    ax[0].set_title("Normalized Slice")
    ax[0].axis("off")
    
    ax[1].imshow(overlay)
    ax[1].set_title("Superpixel Segments (GAT Nodes)")
    ax[1].axis("off")
    plt.tight_layout()
    return fig

# --- Sidebar Controls ---
st.sidebar.title("🧠 Diagnostic Hub")
st.sidebar.markdown("Dual Graph Attention Network (GAT) classification system for brain MRI scans.")

model_status = st.sidebar.empty()
try:
    model, device = load_trained_model()
    model_status.success("Model Status: Online")
except Exception as e:
    model_status.error("Model Status: Offline (Check model.pth path)")

st.sidebar.subheader("Upload MRI Scan")
uploaded_file = st.sidebar.file_uploader("Choose a DICOM/JPG/PNG image", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.info("**Note:** This application provides automated analysis support. Always cross-examine results with clinical diagnostics.")

# --- Main Dashboard ---
st.title("Brain Tumor Multi-Scale Graph Classification")
st.caption("ResNet-50 + CBAM Feature Extraction with Dual-Scale Superpixel GAT Integration")

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    img_tensor = transform_image(raw_image).to(device)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Scan Input & Graph Construction")
        st.image(raw_image, caption="Uploaded MRI Slice", use_container_width=True)
        
        with st.expander("View Graph Node Representation", expanded=False):
            fig_sp = plot_superpixel_overlay(img_tensor)
            st.pyplot(fig_sp)

    with col2:
        st.subheader("2. Diagnostic Output")
        
        analyze_btn = st.button("Run Diagnostic Pipeline", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("Processing feature maps and constructing multi-scale graphs..."):
                with torch.no_grad():
                    log_probs = model(img_tensor)
                    probs = torch.exp(log_probs[0]).cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    confidence = probs[pred_idx] * 100

            # Result Header
            pred_label = CLASS_NAMES[pred_idx]
            if pred_label == "No Tumor":
                st.success(f"**Primary Diagnosis:** {pred_label}")
            else:
                st.warning(f"**Primary Diagnosis:** {pred_label}")

            st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
            st.caption(f"**Class Description:** {CLASS_DESCRIPTIONS[pred_label]}")

            # Confidence Distribution
            st.markdown("---")
            st.subheader("Probability Distribution")
            for i, name in enumerate(CLASS_NAMES):
                score = float(probs[i])
                st.write(f"**{name}** ({score*100:.1f}%)")
                st.progress(score)

else:
    st.info("👈 Please upload an MRI scan image from the sidebar to begin analysis.")
    
    # Placeholder instructions grid
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Backbone Analysis**")
        st.caption("ResNet-50 with CBAM attention channels extracts rich spatial feature maps from the slice.")
    with c2:
        st.markdown("**2. Superpixel Graphing**")
        st.caption("SLIC superpixels cluster features into region-based nodes at fine and coarse scales.")
    with c3:
        st.markdown("**3. Dual GAT Fusion**")
        st.caption("Class-aware Graph Attention Networks weigh inter-region relationships to compute probabilities.")

# import os
# import gdown
# import numpy as np
# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# from model import EnhancedResNet50GAT

# MODEL_PATH = "enhanced_model.pth"

# st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
# st.title("🧠 Brain Tumor Detection & Classification")


# @st.cache_resource
# def load_tumor_model():
#     if not os.path.exists(MODEL_PATH):
#         file_id = "1mhW8fp31-sb3Bv-UYuXTg8bQrARx_6ki"
#         url = f"https://drive.google.com/uc?id={file_id}"
#         gdown.download(url, MODEL_PATH, quiet=False)

#     # Initialize model with 4 classes
#     model = EnhancedResNet50GAT(pretrained=False, num_classes=4)

#     # Load weights
#     state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model


# model = load_tumor_model()

# CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# uploaded_file = st.file_uploader(
#     "Upload MRI Scan", type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded MRI Scan", width=300)

#     img_tensor = transform(image).unsqueeze(0)

#     if st.button("Classify Scan"):
#         with torch.no_grad():
#             log_probs = model(img_tensor)
#             # Convert log_softmax outputs to probabilities using exp()
#             probs = torch.exp(log_probs[0]).numpy()

#         st.write("### Prediction Results")
#         for label, prob in zip(CLASSES, probs):
#             st.progress(float(prob), text=f"{label}: {prob * 100:.1f}%")
