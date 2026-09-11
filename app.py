import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor Detection & Classification")

# Cache model so it stays in memory
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Scan", width=300)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    # Predict
    preds = model.predict(img_array)[0]
    
    st.write("### Prediction Results")
    for label, prob in zip(CLASSES, preds):
        st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
