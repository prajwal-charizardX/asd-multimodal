from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('/workspaces/asd-multimodal/models/autism(1).h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Load and preprocess the image
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img = cv2.imread(img_path)
    resize = tf.image.resize(img, (256, 256))
    prediction = model.predict(np.expand_dims(resize / 255.0, axis=0))

    result = "Not Autistic" if prediction > 0.015 else "Autistic"

    # Render the prediction result along with the image
    return render_template('result.html', image_path=img_path, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
