# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# -----------------------------
# 1️⃣ Load your trained model
# -----------------------------
from tensorflow.keras.models import load_model

model = load_model("skin_cancer_model_final.keras")

# Replace with your actual class names
class_names = ['nv','mel','bkl','bcc','akiec','vasc','df']

# -----------------------------
# 2️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")
st.title("🩺 Skin Cancer Detection App")
st.write("Upload a skin lesion image and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Predicting..."):
        pred_probs = model.predict(img_array)
        pred_class = class_names[np.argmax(pred_probs)]
        confidence = np.max(pred_probs)

    # Display results
    st.markdown(f"### Predicted Class: **{pred_class.upper()}**")
    if confidence > 0.85:
        st.success(f"Confidence: {confidence*100:.2f}% ✅")
    elif confidence > 0.6:
        st.warning(f"Confidence: {confidence*100:.2f}% ⚠️")
    else:
        st.error(f"Confidence: {confidence*100:.2f}% ❌")
    
    st.info("Note: This is a predictive model. Always consult a medical professional for diagnosis.")
