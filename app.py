# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests, os

# ==========================
# 1. Link model & class names (RAW GitHub URL)
# ==========================
MODEL_URL = "https://github.com/zahratalitha/101makanan/blob/894c65365b77386eda858b5856075b3d34672a11/cnn_food101_model_full%20(1).h5"
MODEL_PATH = "cnn_food101_model_full (1).h5"

CLASS_URL = "https://raw.githubusercontent.com/zahratalitha/101makanan/main/class.txt"
CLASS_PATH = "class.txt"

def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            r = requests.get(url, stream=True)
            with open(filename, "wb") as f:
                f.write(r.content)

download_file(MODEL_URL, MODEL_PATH)
download_file(CLASS_URL, CLASS_PATH)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

with open(CLASS_PATH, "r") as f:
    class_names = [line.strip() for line in f]

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

st.title("🍔 Food-101 Image Classifier")
st.write("Upload gambar makanan untuk diprediksi (101 kategori).")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diupload", use_column_width=True)

    # Prediksi
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.subheader(f"🍽️ Prediksi: {predicted_class}")
    st.write(f"✅ Confidence: {confidence:.2f}%")
