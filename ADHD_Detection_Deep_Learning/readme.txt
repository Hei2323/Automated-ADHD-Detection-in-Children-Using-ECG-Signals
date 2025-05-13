# ADHD Classification with Deep Learning

This project aims to classify ADHD using ECG signals with a deep learning model. The main approach uses a **1D-CNN** for feature extraction, and an additional **Score-CAM** method is used to extract high-dimensional features from the CNN for further training with traditional machine learning classifiers.

## Project Structure

### /ECG-data
Age-matched ECG data, ranging from 6 to 12 years old, wereuploaded, including 6-second ECG recordings sampled at 500 Hz, alongwith age, gender, and label. The ADHD to HC ratio was 1:1 for each agegroup, totaling 8,644 samples. The data is provided in numpy format (`ADHD_data.rar`).

### /models
Contains the neural network model definition (`cnn_adhd.py`).

### /feature_extraction
Contains the **Score-CAM** implementation (`score_cam.py`) and the feature extraction methods (`feature_extractor.py`).

### /utils
Contains utilities for data loading and handling (`data_loader.py`).

### train.py
Main script to train the model using **1D-CNN**.

### evaluate.py
Main script for evaluating the trained models. It evaluates both the **CNN** and **CNN + Score-CAM + Machine Learning** approaches.(Note that the training part of the machine learning classifier is also included.)

### requirements.txt
List of Python dependencies required for the project.

### README.md
Project documentation and setup guide.

## Requirements

To run this project, you need to install the following dependencies:

```bash
pip install -r requirements.txt
