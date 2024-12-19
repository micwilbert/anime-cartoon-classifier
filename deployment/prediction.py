import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Google Drive model URL
model_url = "https://drive.google.com/uc?id=1UXFti2uKwcnBW210fGamzoJ9l2hDTkmN"
model_path = "./model.h5"

# Download model
def download_model():
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(model_url, model_path, quiet=False)
        print("Model downloaded.")
    else:
        print("Model already exists.")

download_model()

# Load model
model = load_model(model_path)

# Preprocessing function
def preprocess_image(image, target_size=(380, 380)):
    image = Image.open(image).convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Predict function
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_label = "Anime" if prediction[0][0] < 0.5 else "Cartoon"
    confidence = prediction[0][0] if class_label == "Cartoon" else 1 - prediction[0][0]
    return {"class": class_label, "confidence": float(confidence)}
