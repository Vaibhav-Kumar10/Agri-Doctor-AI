import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Paths
# MODEL_DIR = "model"
# MODEL_FILE = "model\plant_disease_model.keras" 
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_PATH = "model\plant_disease_model.keras"

UPLOAD_FOLDER = "uploads"

# Create uploads folder if not exists
# if not os.path.exists(UPLOAD_FOLDER):
    # os.makedirs(UPLOAD_FOLDER)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model from '{MODEL_PATH}': {e}")
    st.stop()

# Class names - same as your original list
class_names = [
    "Potato___Late_blight",
    "Tomato_healthy",
    "Potato___healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Leaf_Mold",
    "Tomato_Late_blight",
    "Potato___Early_blight",
    "Tomato_Early_blight",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Pepper__bell___Bacterial_spot",
    "Tomato__Target_Spot",
    "Pepper__bell___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Septoria_leaf_spot",
]


def predict(image: Image.Image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    return class_names[class_idx], confidence


# Streamlit UI
st.set_page_config(page_title="🌿 AgriDoctorAI", layout="centered")

st.markdown(
    """
    <style>
        .main {
            background-color: #f4f4f4; 
            padding: 20px; 
            border-radius: 10px; 
            max-width: 600px; 
            margin: auto;
        }
        .title {
            color: #4CAF50; 
            font-size: 2.5em; 
            margin-bottom: 10px; 
            font-weight: bold;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🌿 AgriDoctorAI</div>', unsafe_allow_html=True)
st.write("Upload a leaf image to detect potential plant disease")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        # Save uploaded file to uploads folder
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("🔍 Detecting disease, please wait..."):
            label, confidence = predict(image)

        st.success(f"Prediction: {label}")

        # Proper progress bar for confidence (0 to 100)
        progress_bar = st.progress(0)
        progress_bar.progress(min(int(confidence * 100), 100))

        st.info(f"Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")

# st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
