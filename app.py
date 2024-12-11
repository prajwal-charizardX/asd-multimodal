from flask import Flask, request, render_template, jsonify, url_for
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import cv2
from nilearn import plotting, image
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

app = Flask(__name__)

# Create required folders
UPLOAD_FOLDER = 'static/uploads'
PLOTS_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER

# Load your pre-trained TensorFlow model
model = tf.keras.models.load_model('/workspaces/asd-multimodal/models/autism(1).h5')

# Load the model and scaler
loaded_model = joblib.load('/workspaces/asd-multimodal/models/svm_model_imp.joblib')
loaded_scaler = joblib.load('/workspaces/asd-multimodal/models/scaler_gene.pkl')

@app.route('/predict_gene', methods=['POST'])
def predict_gene():
    try:
        # Get dataset from request
        data = request.json
        dataset = pd.DataFrame(data['dataset'])

        # Scale the input
        scaled_input = loaded_scaler.transform(dataset)

        # Features of interest
        features_of_interest = [
            'number-of-reports',
            'genetic-category_Rare Single Gene Mutation, Syndromic',
            'genetic-category_Rare Single Gene Mutation, Syndromic, Functional',
            'genetic-category_Rare Single Gene Mutation, Syndromic, Genetic Association',
            'chromosome_12', 'chromosome_19', 'genetic-category_Syndromic, Genetic Association',
            'genetic-category_Syndromic', 'genetic-category_Rare Single Gene Mutation, Syndromic, Genetic Association, Functional',
            'chromosome_22', 'chromosome_9', 'chromosome_5', 'chromosome_6', 'chromosome_15', 'chromosome_2', 'chromosome_20',
            'chromosome_14', 'chromosome_X', 'chromosome_7', 'chromosome_10', 'chromosome_16', 'chromosome_21',
            'genetic-category_Genetic Association, Functional', 'chromosome_3', 'chromosome_8', 'gene-score',
            'genetic-category_Rare Single Gene Mutation, Functional', 'genetic-category_Rare Single Gene Mutation, Genetic Association, Functional',
            'genetic-category_Genetic Association', 'genetic-category_Rare Single Gene Mutation, Genetic Association', 'genetic-category_Rare Single Gene Mutation'
        ]

        # Extract the features of interest
        extracted_features = pd.DataFrame(scaled_input, columns=dataset.columns)[features_of_interest]

        # Make predictions
        predictions = loaded_model.predict(extracted_features)
        probabilities = loaded_model.predict_proba(extracted_features)[:, 1]

        # Prepare the response
        response = {
            "prediction": "ASD" if predictions[0] == 1 else "TD",
            "probability": probabilities[0]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Path to atlas file
ATLAS_PATH = os.path.join(os.getcwd(), 'data', 'atlas', 'template_cambridge_basc_multiscale_sym_scale064.nii.gz')

# PyTorch model class
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(2016, 256)
        self.layer_2 = nn.Linear(256, 50)
        self.layer_out = nn.Linear(50, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(50)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

# Initialize PyTorch model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_model = BinaryClassification()
torch_model.load_state_dict(torch.load("/workspaces/asd-multimodal/models/classifier.pt", map_location=device))
torch_model.to(device)
torch_model.eval()

# Load scaler for feature vector normalization
scaler = joblib.load("/workspaces/asd-multimodal/models/scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle image prediction
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Preprocess the image
    img = cv2.imread(img_path)
    resize = tf.image.resize(img, (256, 256))
    prediction = model.predict(np.expand_dims(resize / 255.0, axis=0))

    result = "Not Autistic" if prediction > 0.015 else "Autistic"

    # Render the prediction result along with the image
    return render_template('result-facial.html', image_path=img_path, prediction=result)

@app.route('/visualize', methods=['POST'])
def visualize():
    if 'nii_file' not in request.files:
        return "No file uploaded", 400

    nii_file = request.files['nii_file']
    if nii_file.filename == '':
        return "No file selected", 400

    # Save the uploaded file
    nii_path = os.path.join(UPLOAD_FOLDER, nii_file.filename)
    nii_file.save(nii_path)

    # Generate plots
    try:
        fMRI_plot_path, atlas_overlay_path = generate_plot(nii_path)
    except Exception as e:
        return f"Error generating plots: {str(e)}", 500

    # Render the results
    return render_template(
        'result-brain.html',
        fmri_plot_path=fMRI_plot_path,
        atlas_plot_path=atlas_overlay_path
    )

@app.route('/predict-fmri', methods=['POST'])
def predict_fmri():
    if 'feature-file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['feature-file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Load the feature vector
    try:
        features = np.load(file)['a']
    except Exception as e:
        return jsonify({"error": f"Failed to load feature vector: {str(e)}"}), 400

    # Reshape and scale the features
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    # Convert to PyTorch tensor
    features_tensor = torch.FloatTensor(features).to(device)

    # Predict using the PyTorch model
    with torch.no_grad():
        prediction = torch_model(features_tensor)
        probability = torch.sigmoid(prediction).item()
        predicted_class = int(torch.round(torch.sigmoid(prediction)).item())

    # Return result
    result = {
        "prediction": "ASD" if predicted_class == 0 else "TD",
        "probability": probability
    }
    return jsonify(result)

def generate_plot(nii_path):
    # Load the fMRI image
    fmri_img = image.load_img(nii_path)

    # Plot the fMRI image
    fmri_plot_path = os.path.join(PLOTS_FOLDER, 'fmri_plot.png')
    plotting.plot_epi(fmri_img, title="Uploaded fMRI Image", draw_cross=True, output_file=fmri_plot_path)

    # Load and resample the atlas
    atlas_img = image.load_img(ATLAS_PATH)
    resampled_atlas = resample_to_img(atlas_img, fmri_img, interpolation='nearest')

    # Overlay the atlas on the fMRI image
    atlas_overlay_path = os.path.join(PLOTS_FOLDER, 'atlas_overlay_plot.png')
    plotting.plot_roi(resampled_atlas, bg_img=fmri_img, title="Atlas Overlaid on fMRI", draw_cross=True, output_file=atlas_overlay_path)

    return fmri_plot_path, atlas_overlay_path

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

