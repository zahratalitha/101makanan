import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests, os

MODEL_URL = "https://raw.githubusercontent.com/zahratalitha/101makanan/main/cnn_food101_model_full%20(1).h5"
MODEL_PATH = "cnn_food101_model_full.h5"

CLASS_URL = "https://raw.githubusercontent.com/zahratalitha/101makanan/main/class.txt"
CLASS_PATH = "class.txt"

def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"📥 Downloading {filename}..."):
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
    
st.set_page_config(page_title="🍔 Food-101 Classifier", page_icon="🍴", layout="wide")

st.sidebar.title("📤 Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

st.title("🍔 Food-101 Image Classifier")
st.markdown("Upload gambar makanan untuk diprediksi ke dalam **101 kategori**.")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Gambar yang diupload", use_container_width=True)

    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0]).numpy()

    top5_idx = np.argsort(score)[::-1][:5]
    top5_labels = [class_names[i] for i in top5_idx]
    top5_scores = [score[i] for i in top5_idx]

    predicted_class = top5_labels[0]
    confidence = top5_scores[0] * 100

    st.success(f"🍽️ Prediksi Utama: **{predicted_class}** ({confidence:.2f}%)")

    st.subheader("🔝 Top-5 Prediksi")
    for label, conf in zip(top5_labels, top5_scores):
        st.write(f"**{label}** - {conf*100:.2f}%")
        st.progress(float(conf))

else:
    st.info("⬅️ Silakan upload gambar terlebih dahulu lewat sidebar.")
