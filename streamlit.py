import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import random

# Load the model
MODEL_PATH = r"C:\Users\Destiny\Downloads\Intel Classifier\intel_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Constants
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE = (224, 224)
PRED_FOLDER = r"C:\Users\Destiny\Downloads\Intel Classifier\archive\seg_pred\seg_pred"
CONFUSION_MATRIX_PATH = r"C:\Users\Destiny\Downloads\Intel Classifier\ConfusionMatrix.png"
RESULT_PLOT_PATH = r"C:\Users\Destiny\Downloads\Intel Classifier\Results.png"
BEST_ACCURACY = "91.63%"

# Function to process image
def load_and_prepare_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Function to predict
def predict_image(img_path):
    processed = load_and_prepare_image(img_path)
    prediction = model.predict(processed)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Streamlit App
st.set_page_config(page_title="Intel Image Classifier", layout="centered")

st.title("🧠 Intel Image Classifier")
st.markdown("Classifies images into: **buildings**, **forest**, **glacier**, **mountain**, **sea**, or **street**.")

# --- Buttons and Logic ---
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

with col2:
    if st.button("🎲 Classify Random Image"):
        all_images = [os.path.join(PRED_FOLDER, f) for f in os.listdir(PRED_FOLDER)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if all_images:
            random_path = random.choice(all_images)
            st.image(random_path, caption="Randomly Selected Image", width=300)
            pred = predict_image(random_path)
            st.success(f"Prediction: **{pred}**")

# Uploaded image classification
if uploaded_file:
    image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(image_pil, caption="Uploaded Image", width=300)
    
    # Save uploaded image to temporary location
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    prediction = predict_image("temp.jpg")
    st.success(f"Prediction: **{prediction}**")

# --- Results Buttons ---
st.markdown("---")
if st.button("📊 Show Confusion Matrix"):
    st.image(CONFUSION_MATRIX_PATH, caption="Confusion Matrix", use_column_width=True)

if st.button("📈 Show Training Results"):
    st.image(RESULT_PLOT_PATH, caption=f"Training Results — Best Accuracy: {BEST_ACCURACY}", use_column_width=True)
