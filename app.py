from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'datamind-automl-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained MobileNetV2 model
print("Initializing MobileNetV2 model...")
MODEL_PATH = 'model.h5'
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        raise ValueError("Model not found")
except:
    print("Creating a base MobileNetV2 model...")
    # Create the model using ImageNet weights and save it locally
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=True,
        input_shape=(224, 224, 3)
    )
    # Save the model
    base_model.save(MODEL_PATH)
    model = base_model
    print("Base model created and saved to model.h5")

# Demo ImageNet classes
IMAGENET_CLASSES = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
    'electric_ray', 'stingray', 'cock', 'hen', 'ostrich'
]

def preprocess_image(image):
    """Preprocess image for MobileNetV2 prediction"""
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if doing dataset upload or single image prediction
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image selected'}), 400
            
            # Save and process
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            image = Image.open(file.stream).convert('RGB')
            image.save(filepath)
            
            # Predict
            processed_image = preprocess_image(image)
            # Make sure we use base imagenet decoding if this is MobileNetV2
            predictions = model.predict(processed_image, verbose=0)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
            
            results = []
            for _, class_name, confidence in decoded_predictions:
                results.append({
                    'class': class_name.replace('_', ' ').capitalize(),
                    'confidence': float(confidence) * 100
                })
            
            return jsonify({
                'success': True,
                'results': results,
                'image_path': f"static/uploads/{filename}"
            })
            
        elif 'dataset' in request.files:
            file = request.files['dataset']
            if file.filename == '':
                return jsonify({'error': 'No dataset selected'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"dataset_{filename}")
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'message': f"Dataset '{filename}' successfully uploaded to DataMind cloud storage. Our AutoML engine will begin preprocessing.",
                'filepath': filepath
            })
            
        else:
            return jsonify({'error': 'No valid file provided (image or dataset)'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)