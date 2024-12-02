import os
import sys  # Add this import to use sys.exit()
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import rasterio

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Custom metrics
def recall_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def precision_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Configuration
class CFG:
    img_size = (224, 224)  # Resize images to 224x224
    class_dict = {0: "No Flooding", 1: "Flooding"}

# Load the model
def load_flood_model(model_name):
    try:
        model = load_model(model_name, custom_objects={
            "f1_score": f1_score,
            "recall_m": recall_m,
            "precision_m": precision_m
        }, compile=False)  # Skip compilation during loading
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# Preprocess SAR images (2-channel TIFFs)
# Preprocess RGB images (3-channel JPG/PNG)
def preprocess_rgb_image(image_path, img_size):
    try:
        image = tf.keras.utils.load_img(image_path, target_size=(256, 256))  # Resize to (256, 256)
        image_array = tf.keras.utils.img_to_array(image) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing RGB image: {e}")
        sys.exit(1)

# Preprocess RGB images (3-channel JPG/PNG)
def preprocess_rgb_image(image_path, img_size):
    try:
        image = tf.keras.utils.load_img(image_path, target_size=img_size)
        image_array = tf.keras.utils.img_to_array(image) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing RGB image: {e}")
        sys.exit(1)

# Predict
def predict_flooding(model, image):
    try:
        predictions = model.predict(image)
        print(f"Model raw output: {predictions}")
        class_index = np.argmax(predictions, axis=1)[0]
        return CFG.class_dict[class_index]
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

# Flood detection function
def flood_detection(model_type, model_path, image_path):
    """
    Detects flooding based on the input model and image.

    Parameters:
        model_type (str): 'sar' for SAR models, 'rgb' for RGB models.
        model_path (str): Path to the model file.
        image_path (str): Path to the image file.

    Returns:
        str: Prediction label ('Flooding' or 'No Flooding').
    """
    model = load_flood_model(model_path)

    # Preprocess based on model type
    if model_type == "sar":
        image = preprocess_sar_image(image_path, CFG.img_size)
    elif model_type == "rgb":
        image = preprocess_rgb_image(image_path, CFG.img_size)
    else:
        raise ValueError("Invalid model type. Use 'sar' or 'rgb'.")

    return predict_flooding(model, image)

# Export required functions
__all__ = [
    "flood_detection",
    "preprocess_rgb_image",
    "load_flood_model",
    "preprocess_sar_image",
    "predict_flooding"
]

