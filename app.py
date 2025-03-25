from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from flask_cors import CORS
from scipy.stats import entropy

# Define class names
class_names = [
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy', 'Potato Early Blight', 
    'Potato Late Blight', 'Potato Healthy', 'Tomato Bacterial Spot', 'Tomato Early Blight', 
    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 
    'Tomato Spider Mites', 'Tomato Target Spot', 
    'Tomato Yellow Leaf Curl', 'Tomato Mosaic Virus', 'Tomato Healthy'
]

# Define paths
BASE_DIR = "./"  # Base directory
FRONTEND_DIR = os.path.join(BASE_DIR)  # This is where Plant.html is located

app = Flask(__name__, static_folder=FRONTEND_DIR, template_folder=FRONTEND_DIR)

# **Move CORS(app) here, after defining 'app'**
CORS(app)  # Allow cross-origin requests

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR,"mobilenet_finetuned_best.h5")  # Path to the trained model

# Check if model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image)  # Convert to numpy array

    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Serve the frontend file
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "Plant1.html")

# Predict function
@app.route('/predict', methods=['POST'])

def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    print(file)
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    confidence = float(np.max(prediction))
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    pred_entropy = entropy(prediction[0])

    print(f"Confidence: {confidence}, Predicted Class: {predicted_class}, Entropy: {pred_entropy}")

    # Confidence and entropy-based rejection
    if confidence < 0.8 or pred_entropy > 1.5:
        return jsonify({"prediction": "Unknown", "confidence": confidence})

    disease_name = class_names[predicted_class]
    return jsonify({"prediction": disease_name, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)

