# Multivariate-Antibiotic-Prediction-Model-Custom-cnn-
Multivariate Antibiotic Prediction Model (Custom cnn)
Due to the large number of datasets, we have decided to upload the datasets to
Link: https://pan.baidu.com/s/14alwxrGsbZtbCLJeqGbG_Q?pwd=rb88 extraction code: rb88


# CNN for Multi-Metric Image Regression
A TensorFlow-Keras implementation of a CNN model for regression tasks using image data (supports multiple regression targets, e.g., age, height, weight from facial images).

## Features
- **Dynamic Metric Support**: Auto-detects regression targets from Excel labels (no need to modify model output layer).
- **Hyperparameter Search**: Uses `RandomizedSearchCV` to optimize learning rate, conv filters, etc.
- **Visualization Tools**: 
  - Intermediate convolutional layer activation maps
  - Grad-CAM for feature importance (which parts of the image influence predictions)
  - True vs Predicted value scatter plots
- **Robust Validation**: K-fold cross-validation + train/test split for reliable performance evaluation.

## Environment Setup
Install required dependencies:
```bash
pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow openpyxl


Python 3.8+
TensorFlow 2.4+ (for .keras model saving format)
OpenPyXL (for reading Excel files)
