from keras.models import load_model
from PIL import Image, ImageOps  
import numpy as np

def load_keras_model(model_path):
    # Load the Keras model
    model = load_model(model_path, compile=False)
    return model

def load_labels(labels_path):
    # Load the class labels
    with open(labels_path, "r") as f:
        class_names = f.readlines()
    return class_names

def preprocess_image(image_path, img_size=(224, 224)):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, img_size[0], img_size[1], 3), dtype=np.float32)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, img_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    return data

def predict_image(model, data):
    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index] # Confidence of predicted class
    confidence_score = prediction[0]
    return index, confidence_score