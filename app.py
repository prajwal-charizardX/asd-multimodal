from flask import Flask, request, render_template, jsonify, url_for
import tensorflow as tf
import numpy as np
import os
import cv2
from nilearn import plotting, image
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt

app = Flask(__name__)

# Create required folders
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER

# Load your pre-trained model
model = tf.keras.models.load_model('/workspaces/asd-multimodal/models/autism(1).h5')

# Path to atlas file
ATLAS_PATH = os.path.join(os.getcwd(), 'data', 'atlas', 'template_cambridge_basc_multiscale_sym_scale064.nii.gz')

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
    app.run(debug=True)
