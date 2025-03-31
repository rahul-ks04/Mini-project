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


treatment_info = {
    'Pepper Bell Bacterial Spot': "Symptoms: Small water-soaked lesions on leaves and fruits that turn brown.<br>"
                                  "Causes: Bacterial infection spread through infected seeds and water.<br>"
                                  "Impact: Reduces fruit quality and weakens plant health.<br>"
                                  "Treatment: Use copper-based fungicides, plant disease-resistant varieties, and ensure proper irrigation practices.",
    'Pepper Bell Healthy': "Info: The plant shows no signs of disease.<br>"
                           "Causes: Optimal care and healthy environment.<br>"
                           "Impact: Thriving growth and high yield.<br>"
                           "Treatment: Maintain balanced nutrients, adequate watering, and pest control as preventive measures.",
    'Potato Early Blight': "Symptoms: Dark spots with yellow halos on older leaves.<br>"
                           "Causes: Fungus Alternaria solani thrives in warm, humid conditions.<br>"
                           "Impact: Weakens the plant and reduces tuber growth.<br>"
                           "Treatment: Apply fungicides like Chlorothalonil, remove affected leaves, and practice crop rotation.",
    'Potato Late Blight': "Symptoms: Large dark spots on leaves with white fungal growth.<br>"
                          "Causes: Water mold Phytophthora infestans in wet conditions.<br>"
                          "Impact: Can destroy entire crops and cause tuber rot.<br>"
                          "Treatment: Use fungicides like Mancozeb, improve drainage, and plant resistant varieties.",
    'Potato Healthy': "Info: The plant is free from diseases and pests.<br>"
                      "Causes: Good agricultural practices.<br>"
                      "Impact: Healthy plants with maximum yield.<br>"
                      "Treatment: Continue proper watering, fertilization, and preventive care.",
    'Tomato Bacterial Spot': "Symptoms: Water-soaked lesions on leaves and fruits.<br>"
                             "Causes: Bacteria spread by infected seeds or water splashes.<br>"
                             "Impact: Reduces fruit quality and yield.<br>"
                             "Treatment: Disinfect tools, apply copper fungicides, and plant resistant seeds.",
    'Tomato Early Blight': "Symptoms: Dark spots with concentric rings on leaves, wilting of older leaves.<br>"
                           "Causes: Fungus Alternaria solani.<br>"
                           "Impact: Reduces photosynthesis and stunts plant growth.<br>"
                           "Treatment: Use fungicides like Mancozeb, remove infected leaves, and ensure proper spacing.",
    'Tomato Late Blight': "Symptoms: Large dark spots on leaves and fruits with a white fungal halo.<br>"
                          "Causes: Water mold Phytophthora infestans in cool, wet conditions.<br>"
                          "Impact: Severe damage to plants and fruits.<br>"
                          "Treatment: Apply fungicides like Mancozeb, improve ventilation, and avoid overhead watering.",
    'Tomato Leaf Mold': "Symptoms: Yellow spots on upper leaf surfaces and mold on the underside.<br>"
                        "Causes: Fungus Cladosporium in high humidity.<br>"
                        "Impact: Weakens the plant and reduces fruit production.<br>"
                        "Treatment: Improve air circulation, reduce humidity, and apply fungicides.",
    'Tomato Septoria Leaf Spot': "Symptoms: Small, circular spots with gray centers on lower leaves.<br>"
                                 "Causes: Fungus Septoria lycopersici spread by splashing water.<br>"
                                 "Impact: Premature leaf drop and reduced photosynthesis.<br>"
                                 "Treatment: Remove infected leaves and use fungicides like Chlorothalonil.",
    'Tomato Spider Mites': "Symptoms: Yellow specks on leaves, webbing, and browning leaves.<br>"
                           "Causes: Spider mites feeding on leaf tissues in dry conditions.<br>"
                           "Impact: Weakens plant health and lowers yield.<br>"
                           "Treatment: Use insecticidal soap or neem oil to control infestations.",
    'Tomato Target Spot': "Symptoms: Round dark spots with concentric rings on leaves and fruits.<br>"
                          "Causes: Fungus Corynespora cassiicola.<br>"
                          "Impact: Reduces fruit quality and weakens plants.<br>"
                          "Treatment: Apply fungicides, maintain proper spacing, and ensure good ventilation.",
    'Tomato Yellow Leaf Curl': "Symptoms: Yellowing and curling leaves, stunted plant growth.<br>"
                               "Causes: Virus spread by whiteflies.<br>"
                               "Impact: Reduced plant vigor and fruit production.<br>"
                               "Treatment: Control whiteflies using insecticides and plant disease-resistant varieties.",
    'Tomato Mosaic Virus': "Symptoms: Mottled yellow and green patterns on leaves; distorted fruits.<br>"
                           "Causes: Virus spread through infected tools or plants.<br>"
                           "Impact: Reduces fruit quality and growth.<br>"
                           "Treatment: Remove infected plants, disinfect tools, and use virus-resistant seeds.",
    'Tomato Healthy': "Info: Plant shows no signs of disease.<br>"
                      "Causes: Optimal growing conditions.<br>"
                      "Impact: Thriving growth and high yield.<br>"
                      "Treatment: Regular care, pest control, and nutrient balance will keep the plant thriving."
}

# Define paths
BASE_DIR = "./"
FRONTEND_DIR = os.path.join(BASE_DIR)  # This is where Plant.html is located

app = Flask(__name__, static_folder=FRONTEND_DIR, template_folder=FRONTEND_DIR)

# **Move CORS(app) here, after defining 'app'**
CORS(app)  # Allow cross-origin requests

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR,"mobilenet_finetuned_best2.h5")

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
    return send_from_directory(FRONTEND_DIR, "plant1.html")

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
    treatment = treatment_info.get(disease_name, "No treatment information available for this disease.")

    return jsonify({"prediction": disease_name, "confidence": confidence, "treatment": treatment})

if __name__ == '__main__':
    app.run(debug=True)
