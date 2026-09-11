import os
import gdown
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Import model helper for GAT graph construction
from model import EnhancedResNet50GAT, build_multiscale_graphs

MODEL_PATH = "enhanced_model.pth"

st.set_page_config(
    page_title="Brain Tumor Detection & Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# EXPLAINABILITY VISUALIZATION HELPERS
# ============================================================

class GradCAMStreamlit:
    def __init__(self, model):
        self.model = model
        self.act = None
        self.grad = None

    def generate(self, img_tensor, target_class=None):
        self.model.eval()
        
        # Ensure gradients are explicitly enabled for Grad-CAM computation
        with torch.enable_grad():
            img_t = img_tensor.clone().detach().requires_grad_(True)

            # Target target layer: layer4 backbone
            target_layer = self.model.backbone.layer4

            # Define forward hook
            def forward_hook(module, input, output):
                self.act = output

            # Define tensor-level backward hook (more reliable than module backward hook)
            def backward_hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            # Register dynamic hooks
            h_fw = target_layer.register_forward_hook(forward_hook)
            h_bw = target_layer.register_full_backward_hook(backward_hook)

            # Forward pass
            out = self.model(img_t)
            if target_class is None:
                target_class = out.argmax(1).item()

            # Zero existing gradients and trigger backward pass
            self.model.zero_grad()
            score = out[0, target_class]
            score.backward(retain_graph=True)

            # Remove hooks immediately after execution
            h_fw.remove()
            h_bw.remove()

            # Safety fallback: If backward hook didn't capture gradients, extract manually from act
            if self.grad is None and self.act is not None and self.act.grad is not None:
                self.grad = self.act.grad

            # If grad is still None, raise a clean exception with context
            if self.grad is None:
                raise RuntimeError("Grad-CAM failed to capture gradients from backbone layer4.")

            # Compute feature weights and activation map
            w = self.grad.detach().mean([2, 3], keepdim=True)
            act = self.act.detach()
            cam = F.relu((w * act).sum(1, keepdim=True))
            cam = F.interpolate(cam, (224, 224), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            
            # Min-Max normalize
            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            return cam, target_class

def visualize_gradcam_fig(model, raw_image, img_tensor, target_class, class_name):
    gcam = GradCAMStreamlit(model)
    hm, pc = gcam.generate(img_tensor, target_class)

    inp = np.array(raw_image.resize((224, 224))) / 255.0
    overlay = 0.5 * inp + 0.5 * plt.cm.jet(hm)[:, :, :3]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(inp)
    ax[0].set_title("Original Image")
    ax[1].imshow(hm, cmap='jet')
    ax[1].set_title("Grad-CAM Heatmap")
    ax[2].imshow(overlay)
    ax[2].set_title(f"Overlay - Pred: {class_name}")

    for a in ax:
        a.axis('off')
    plt.tight_layout()
    return fig, hm

def visualize_gat_attention_fig(model, img_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)

        fm = model.backbone(img_tensor)
        fg, _ = build_multiscale_graphs(img_tensor[0].cpu(), fm[0].cpu(), 0)
        fg = fg.to(device)

        g1 = model.gat.fg1
        H, C = g1.heads, g1.out_channels
        xp = g1.lin(fg.x).view(-1, H, C)
        as_ = (xp * g1.att_src).sum(-1)
        ad = (xp * g1.att_dst).sum(-1)

        ni = torch.zeros(fg.x.size(0), device=device)
        ei = fg.edge_index
        for e in range(ei.size(1)):
            s, d = ei[0, e].item(), ei[1, e].item()
            ni[d] += abs((as_[s] + ad[d]).mean().item())

        ni = ni.cpu().numpy()
        ni = (ni - ni.min()) / (ni.max() - ni.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(range(len(ni)), ni, c=ni, cmap='hot', s=100, edgecolors='black')
        ax.set_xlabel("Node (Superpixel)")
        ax.set_ylabel("Attention")
        ax.set_title("GAT Node Attention (Fine Scale)")
        plt.colorbar(sc, label="Importance")
        plt.tight_layout()
        return fig

# ============================================================
# PAGE HEADER
# ============================================================

st.title("🧠 Brain Tumor Detection & Classification")
st.markdown(
    """
    ### Deep Learning-Based MRI Image Classification

    Upload a **brain MRI image** in JPG, JPEG, or PNG format.
    The trained deep learning model will analyze the image and
    classify it into one of four categories:

    **Glioma • Meningioma • No Tumor • Pituitary**
    """
)

st.divider()

# ============================================================
# IMPORTANT INFORMATION / INSTRUCTIONS
# ============================================================

st.info(
    """
    **How to use this application**

    1. Upload a brain MRI scan using the uploader below.
    2. Make sure the uploaded image is clear and relevant to brain MRI analysis.
    3. Preview the uploaded image.
    4. Click **Classify Scan**.
    5. The Deep Learning model will generate the predicted class and probability
       distribution for all four categories.

    The image is automatically resized to **224 × 224 pixels** before
    being passed to the trained model.
    """
)

with st.expander("📌 Image Requirements & Guidelines"):
    st.markdown(
        """
        **For the best demonstration results:**

        - Upload a brain MRI image.
        - Supported formats: **JPG, JPEG, PNG**
        - Use a clear, properly visible MRI image.
        - Avoid extremely blurred, corrupted, or unrelated images.
        - The model expects an RGB image and performs the required
          preprocessing automatically.
        - The model was trained for four-class brain tumor classification.

        **Classification categories:**

        | Class | Meaning |
        |---|---|
        | 🧠 Glioma | Tumor arising from glial cells of the brain or spinal cord. |
        | 🧠 Meningioma | Tumor arising from the protective membranes surrounding the brain and spinal cord. |
        | ✅ No Tumor | MRI classified by the model as showing no detectable brain tumor. |
        | 🧠 Pituitary | Tumor arising in the pituitary gland at the base of the brain. |
        """
    )

# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

with st.expander("⚠️ Important Medical Disclaimer"):
    st.warning(
        """
        **Research / Demonstration Use Only**

        This application is an academic/research demonstration of a
        deep learning model for brain MRI image classification.

        The prediction generated by this application **must not be
        considered a medical diagnosis or a substitute for evaluation
        by a qualified radiologist or medical professional**.

        Model predictions may be affected by image quality, image type,
        preprocessing, dataset characteristics, and other factors.
        Always consult a qualified healthcare professional for clinical
        interpretation.
        """
    )

st.divider()

# ============================================================
# MODEL LOADING
# DO NOT CHANGE THIS SECTION
# ============================================================

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

# ============================================================
# CLASS INFORMATION
# ============================================================

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload Brain MRI Scan")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG brain MRI image."
)

# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.success("MRI image uploaded successfully.")

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        st.subheader("🔍 Uploaded MRI Scan")

        st.image(
            image,
            caption="Input MRI Scan",
            width=400
        )

        st.caption(
            f"Image size: {image.size[0]} × {image.size[1]} pixels"
        )

        st.divider()

        # ----------------------------------------------------
        # CLASSIFICATION BUTTON
        # ----------------------------------------------------

        st.subheader("🤖 Deep Learning Classification")

        st.write(
            "Click the button below to run the trained deep learning model "
            "on the uploaded MRI image."
        )

        if st.button(
            "🔬 Classify Scan",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing MRI scan..."):

                # Preprocess image
                img_tensor = transform(image).unsqueeze(0)

                # Model inference
                with torch.no_grad():

                    log_probs = model(img_tensor)

                    # Convert log-softmax outputs to probabilities
                    probs = torch.exp(log_probs[0]).numpy()

            # ------------------------------------------------
            # FIND PREDICTED CLASS
            # ------------------------------------------------

            predicted_index = int(np.argmax(probs))
            predicted_class = CLASSES[predicted_index]
            confidence = float(probs[predicted_index])

            # ------------------------------------------------
            # DISPLAY MAIN RESULT
            # ------------------------------------------------

            st.divider()

            st.subheader("📊 Prediction Results")

            if predicted_class == "notumor":
                display_class = "No Tumor"
            else:
                display_class = predicted_class.capitalize()

            st.success(
                f"### Predicted Class: {display_class}"
            )

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence * 100:.2f}%"
            )

            # ------------------------------------------------
            # CONFIDENCE MESSAGE
            # ------------------------------------------------

            if confidence >= 0.90:

                st.success(
                    "High model confidence for the predicted class."
                )

            elif confidence >= 0.70:

                st.info(
                    "Moderate model confidence. The prediction should "
                    "be interpreted cautiously."
                )

            else:

                st.warning(
                    "Low model confidence. The model is relatively "
                    "uncertain about this classification."
                )

            # ------------------------------------------------
            # ALL CLASS PROBABILITIES
            # ------------------------------------------------

            st.subheader("📈 Class Probability Distribution")

            for label, prob in zip(CLASSES, probs):

                if label == "notumor":
                    display_label = "No Tumor"
                else:
                    display_label = label.capitalize()

                st.progress(
                    float(prob),
                    text=f"{display_label}: {prob * 100:.2f}%"
                )

            # ------------------------------------------------
            # RESULT INTERPRETATION
            # ------------------------------------------------

            st.subheader("📝 Result Interpretation")

            if predicted_class == "glioma":

                st.write(
                    """
                    The model classified the uploaded MRI image as
                    **Glioma** with the highest predicted probability.
                    """
                )

            elif predicted_class == "meningioma":

                st.write(
                    """
                    The model classified the uploaded MRI image as
                    **Meningioma** with the highest predicted probability.
                    """
                )

            elif predicted_class == "pituitary":

                st.write(
                    """
                    The model classified the uploaded MRI image as
                    **Pituitary tumor** with the highest predicted probability.
                    """
                )

            else:

                st.write(
                    """
                    The model classified the uploaded MRI image as
                    **No Tumor** with the highest predicted probability.
                    """
                )

            # ------------------------------------------------
            # EXPLAINABILITY VISUALIZATIONS
            # ------------------------------------------------

            st.divider()
            st.subheader("🔬 Visual Explainability & Model Attention")

            with st.spinner("Generating Grad-CAM and GAT attention visualizations..."):
                fig_cam, _ = visualize_gradcam_fig(model, image, img_tensor, predicted_index, display_class)
                fig_gat = visualize_gat_attention_fig(model, img_tensor)

            st.markdown("#### Grad-CAM Spatial Heatmap")
            st.pyplot(fig_cam)

            st.markdown("#### Graph Attention Network (GAT) Node Weights")
            st.pyplot(fig_gat)

            # ------------------------------------------------
            # DEMO DISCLAIMER
            # ------------------------------------------------

            st.warning(
                """
                **Important:** This result represents the prediction
                of the trained deep learning model and should not be
                interpreted as a clinical diagnosis.
                """
            )

    except Exception as e:

        st.error(
            "Unable to process the uploaded image."
        )

        st.exception(e)

else:

    # ========================================================
    # INITIAL STATE
    # ========================================================

    st.info(
        "👆 Upload a brain MRI image above to begin classification."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 Deep Learning Based Brain Tumor Detection & Classification"
)

st.caption(
    "Model: Enhanced ResNet50-CBAM + Multi-Scale Class-Aware GAT"
)

st.caption(
    "For academic/research demonstration purposes only."
)

# import os
# import gdown
# import numpy as np
# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# from model import EnhancedResNet50GAT

# MODEL_PATH = "enhanced_model.pth"

# st.set_page_config(
#     page_title="Brain Tumor Detection & Classification",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # ============================================================
# # PAGE HEADER
# # ============================================================

# st.title("🧠 Brain Tumor Detection & Classification")
# st.markdown(
#     """
#     ### Deep Learning-Based MRI Image Classification

#     Upload a **brain MRI image** in JPG, JPEG, or PNG format.
#     The trained deep learning model will analyze the image and
#     classify it into one of four categories:

#     **Glioma • Meningioma • No Tumor • Pituitary**
#     """
# )

# st.divider()

# # ============================================================
# # IMPORTANT INFORMATION / INSTRUCTIONS
# # ============================================================

# st.info(
#     """
#     **How to use this application**

#     1. Upload a brain MRI scan using the uploader below.
#     2. Make sure the uploaded image is clear and relevant to brain MRI analysis.
#     3. Preview the uploaded image.
#     4. Click **Classify Scan**.
#     5. The Deep Learning model will generate the predicted class and probability
#        distribution for all four categories.

#     The image is automatically resized to **224 × 224 pixels** before
#     being passed to the trained model.
#     """
# )

# with st.expander("📌 Image Requirements & Guidelines"):
#     st.markdown(
#         """
#         **For the best demonstration results:**

#         - Upload a brain MRI image.
#         - Supported formats: **JPG, JPEG, PNG**
#         - Use a clear, properly visible MRI image.
#         - Avoid extremely blurred, corrupted, or unrelated images.
#         - The model expects an RGB image and performs the required
#           preprocessing automatically.
#         - The model was trained for four-class brain tumor classification.

#         **Classification categories:**

#         | Class | Meaning |
#         |---|---|
#         | 🧠 Glioma | Tumor arising from glial cells of the brain or spinal cord. |
#         | 🧠 Meningioma | Tumor arising from the protective membranes surrounding the brain and spinal cord. |
#         | ✅ No Tumor | MRI classified by the model as showing no detectable brain tumor. |
#         | 🧠 Pituitary | Tumor arising in the pituitary gland at the base of the brain. |
#         """
#     )

# # ============================================================
# # MEDICAL DISCLAIMER
# # ============================================================

# with st.expander("⚠️ Important Medical Disclaimer"):
#     st.warning(
#         """
#         **Research / Demonstration Use Only**

#         This application is an academic/research demonstration of a
#         deep learning model for brain MRI image classification.

#         The prediction generated by this application **must not be
#         considered a medical diagnosis or a substitute for evaluation
#         by a qualified radiologist or medical professional**.

#         Model predictions may be affected by image quality, image type,
#         preprocessing, dataset characteristics, and other factors.
#         Always consult a qualified healthcare professional for clinical
#         interpretation.
#         """
#     )

# st.divider()

# # ============================================================
# # MODEL LOADING
# # DO NOT CHANGE THIS SECTION
# # ============================================================

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

# # ============================================================
# # CLASS INFORMATION
# # ============================================================

# CLASSES = [
#     "glioma",
#     "meningioma",
#     "notumor",
#     "pituitary"
# ]

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

# # ============================================================
# # IMAGE UPLOAD
# # ============================================================

# st.subheader("📤 Upload Brain MRI Scan")

# uploaded_file = st.file_uploader(
#     "Choose an MRI image",
#     type=["jpg", "jpeg", "png"],
#     help="Upload a JPG, JPEG, or PNG brain MRI image."
# )

# # ============================================================
# # PROCESS UPLOADED IMAGE
# # ============================================================

# if uploaded_file is not None:

#     try:

#         image = Image.open(uploaded_file).convert("RGB")

#         st.success("MRI image uploaded successfully.")

#         # ----------------------------------------------------
#         # IMAGE PREVIEW
#         # ----------------------------------------------------

#         st.subheader("🔍 Uploaded MRI Scan")

#         st.image(
#             image,
#             caption="Input MRI Scan",
#             width=400
#         )

#         st.caption(
#             f"Image size: {image.size[0]} × {image.size[1]} pixels"
#         )

#         st.divider()

#         # ----------------------------------------------------
#         # CLASSIFICATION BUTTON
#         # ----------------------------------------------------

#         st.subheader("🤖 Deep Learning Classification")

#         st.write(
#             "Click the button below to run the trained deep learning model "
#             "on the uploaded MRI image."
#         )

#         if st.button(
#             "🔬 Classify Scan",
#             type="primary",
#             use_container_width=True
#         ):

#             with st.spinner("Analyzing MRI scan..."):

#                 # Preprocess image
#                 img_tensor = transform(image).unsqueeze(0)

#                 # Model inference
#                 with torch.no_grad():

#                     log_probs = model(img_tensor)

#                     # Convert log-softmax outputs to probabilities
#                     probs = torch.exp(log_probs[0]).numpy()

#             # ------------------------------------------------
#             # FIND PREDICTED CLASS
#             # ------------------------------------------------

#             predicted_index = int(np.argmax(probs))
#             predicted_class = CLASSES[predicted_index]
#             confidence = float(probs[predicted_index])

#             # ------------------------------------------------
#             # DISPLAY MAIN RESULT
#             # ------------------------------------------------

#             st.divider()

#             st.subheader("📊 Prediction Results")

#             if predicted_class == "notumor":
#                 display_class = "No Tumor"
#             else:
#                 display_class = predicted_class.capitalize()

#             st.success(
#                 f"### Predicted Class: {display_class}"
#             )

#             st.metric(
#                 label="Prediction Confidence",
#                 value=f"{confidence * 100:.2f}%"
#             )

#             # ------------------------------------------------
#             # CONFIDENCE MESSAGE
#             # ------------------------------------------------

#             if confidence >= 0.90:

#                 st.success(
#                     "High model confidence for the predicted class."
#                 )

#             elif confidence >= 0.70:

#                 st.info(
#                     "Moderate model confidence. The prediction should "
#                     "be interpreted cautiously."
#                 )

#             else:

#                 st.warning(
#                     "Low model confidence. The model is relatively "
#                     "uncertain about this classification."
#                 )

#             # ------------------------------------------------
#             # ALL CLASS PROBABILITIES
#             # ------------------------------------------------

#             st.subheader("📈 Class Probability Distribution")

#             for label, prob in zip(CLASSES, probs):

#                 if label == "notumor":
#                     display_label = "No Tumor"
#                 else:
#                     display_label = label.capitalize()

#                 st.progress(
#                     float(prob),
#                     text=f"{display_label}: {prob * 100:.2f}%"
#                 )

#             # ------------------------------------------------
#             # RESULT INTERPRETATION
#             # ------------------------------------------------

#             st.subheader("📝 Result Interpretation")

#             if predicted_class == "glioma":

#                 st.write(
#                     """
#                     The model classified the uploaded MRI image as
#                     **Glioma** with the highest predicted probability.
#                     """
#                 )

#             elif predicted_class == "meningioma":

#                 st.write(
#                     """
#                     The model classified the uploaded MRI image as
#                     **Meningioma** with the highest predicted probability.
#                     """
#                 )

#             elif predicted_class == "pituitary":

#                 st.write(
#                     """
#                     The model classified the uploaded MRI image as
#                     **Pituitary tumor** with the highest predicted probability.
#                     """
#                 )

#             else:

#                 st.write(
#                     """
#                     The model classified the uploaded MRI image as
#                     **No Tumor** with the highest predicted probability.
#                     """
#                 )

#             # ------------------------------------------------
#             # DEMO DISCLAIMER
#             # ------------------------------------------------

#             st.warning(
#                 """
#                 **Important:** This result represents the prediction
#                 of the trained deep learning model and should not be
#                 interpreted as a clinical diagnosis.
#                 """
#             )

#     except Exception as e:

#         st.error(
#             "Unable to process the uploaded image."
#         )

#         st.exception(e)

# else:

#     # ========================================================
#     # INITIAL STATE
#     # ========================================================

#     st.info(
#         "👆 Upload a brain MRI image above to begin classification."
#     )

# # ============================================================
# # FOOTER
# # ============================================================

# st.divider()

# st.caption(
#     "🧠 Deep Learning Based Brain Tumor Detection & Classification"
# )

# st.caption(
#     "Model: Enhanced ResNet50-CBAM + Multi-Scale Class-Aware GAT"
# )

# st.caption(
#     "For academic/research demonstration purposes only."
# )

# # import os
# # import gdown
# # import numpy as np
# # import streamlit as st
# # import torch
# # import torchvision.transforms as transforms
# # from PIL import Image

# # from model import EnhancedResNet50GAT

# # MODEL_PATH = "enhanced_model.pth"

# # st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
# # st.title("🧠 Brain Tumor Detection & Classification")


# # @st.cache_resource
# # def load_tumor_model():
# #     if not os.path.exists(MODEL_PATH):
# #         file_id = "1mhW8fp31-sb3Bv-UYuXTg8bQrARx_6ki"
# #         url = f"https://drive.google.com/uc?id={file_id}"
# #         gdown.download(url, MODEL_PATH, quiet=False)

# #     # Initialize model with 4 classes
# #     model = EnhancedResNet50GAT(pretrained=False, num_classes=4)

# #     # Load weights
# #     state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
# #     model.load_state_dict(state_dict)
# #     model.eval()
# #     return model


# # model = load_tumor_model()

# # CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # ])

# # uploaded_file = st.file_uploader(
# #     "Upload MRI Scan", type=["jpg", "jpeg", "png"]
# # )

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file).convert("RGB")
# #     st.image(image, caption="Uploaded MRI Scan", width=300)

# #     img_tensor = transform(image).unsqueeze(0)

# #     if st.button("Classify Scan"):
# #         with torch.no_grad():
# #             log_probs = model(img_tensor)
# #             # Convert log_softmax outputs to probabilities using exp()
# #             probs = torch.exp(log_probs[0]).numpy()

# #         st.write("### Prediction Results")
# #         for label, prob in zip(CLASSES, probs):
# #             st.progress(float(prob), text=f"{label}: {prob * 100:.1f}%")
