# DNA Splice-Junction Classification

A machine learning and deep learning project for classifying DNA sequences into splice-junction categories.

## Project Overview

This project uses the UCI Molecular Biology Splice Junction dataset to investigate whether DNA sequences can be classified as:

- **EI** — exon-intron splice junction
- **IE** — intron-exon splice junction
- **N** — neither

The project combines data analysis, classical machine learning, deep learning, and model deployment.

## Objectives

- Explore and understand DNA sequence data.
- Preprocess and encode DNA sequences.
- Build classical machine-learning baselines.
- Train a 1D Convolutional Neural Network (CNN).
- Compare model performance using standard classification metrics.
- Build a web application for DNA sequence prediction.

## Models

The project includes:

1. Logistic Regression
2. Random Forest
3. 1D Convolutional Neural Network (CNN)

## Feature Representation

Two approaches are used:

- **3-mer frequency features** for classical machine-learning models.
- **One-hot encoded DNA sequences** for the CNN.

## Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Actual performance results will be added after the models are trained and evaluated.

## Streamlit Application

The project includes a Streamlit web application that allows users to enter a DNA sequence and receive a predicted splice-junction class together with the model's probability estimates.

## Project Structure

```text
dna-splice-junction-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_deep_learning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── models/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE

 Dataset

UCI Molecular Biology (Splice-junction Gene Sequences)

Dataset ID: 69

The dataset contains DNA sequences associated with splice-junction classification.

 Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Jupyter Notebook
* Streamlit

 Project Workflow

DNA Dataset
     ↓
Data Exploration
     ↓
Preprocessing
     ↓
Feature Engineering
     ↓
Classical ML
     ↓
Deep Learning
     ↓
Model Evaluation
     ↓
Streamlit Deployment

 Disclaimer

This project is an educational machine-learning project and is not intended for medical or clinical use.
