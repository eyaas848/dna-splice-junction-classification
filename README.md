DNA Splice-Junction Classification

A machine-learning and deep-learning project for classifying DNA splice-junction sequences into three biological classes: exon-intron (EI), intron-exon (IE), and neither (N).

The project combines classical machine-learning methods, k-mer feature engineering, a 1D convolutional neural network, model evaluation, and a Streamlit web application.

Project Overview

Splice junctions are important regions in DNA where introns and exons meet during gene processing.

This project uses the UCI Molecular Biology Splice Junction dataset to classify 60-base DNA sequences into three categories:

* EI — exon-intron boundary
* IE — intron-exon boundary
* N — neither

The goal is to compare classical machine-learning approaches with a deep-learning approach for DNA sequence classification.

Objectives

* Explore and preprocess DNA sequence data
* Convert nucleotide sequences into machine-learning representations
* Engineer 3-mer frequency features
* Train classical machine-learning models
* Train a 1D convolutional neural network
* Compare model performance
* Analyze classification errors using confusion matrices
* Deploy the trained CNN through a Streamlit application

Models

The project evaluates three models:

1. Logistic Regression

Uses normalized 3-mer frequency features as input.

2. Random Forest

Uses the same 3-mer representation while learning nonlinear decision boundaries through an ensemble of decision trees.

3. 1D Convolutional Neural Network

Uses one-hot encoded DNA sequences and convolutional layers to learn local sequence patterns directly from the nucleotide sequence.

Results

Evaluation was performed on a held-out test set containing 20% of the dataset.

Model	Accuracy	Weighted F1
Logistic Regression	54.70%	45.98%
Random Forest	68.97%	67.49%
1D CNN	81.50%	81.26%

The 1D CNN achieved the highest test-set accuracy and weighted F1 score among the three evaluated models.

These results correspond to the current train/test split and should not be interpreted as a guarantee of performance on other datasets.

Evaluation

The project generates:

* CNN training and validation accuracy
* Logistic Regression confusion matrix
* Random Forest confusion matrix
* 1D CNN confusion matrix
* Classification metrics
* Saved model evaluation results

CNN Training Accuracy

Logistic Regression Confusion Matrix

Random Forest Confusion Matrix

1D CNN Confusion Matrix

Feature Representation

Classical Machine Learning

The classical models use normalized 3-mer frequency features.

A 3-mer is a sequence of three consecutive nucleotides, such as:

ATG
CGA
TTC

All possible combinations of A, C, G, and T are represented as features.

Deep Learning

The CNN uses one-hot encoding to represent each nucleotide:

A → [1, 0, 0, 0]
C → [0, 1, 0, 0]
G → [0, 0, 1, 0]
T → [0, 0, 0, 1]

This produces a sequence representation suitable for convolutional neural networks.

Streamlit Application

The project includes an interactive Streamlit application.

Users can enter a 60-base DNA sequence and receive:

* Predicted class
* Prediction confidence
* Probability for each class

The application uses the trained 1D CNN model.

To launch the application locally:

python -m streamlit run app.py

Project Structure

dna-splice-junction-classification/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
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
│   ├── cnn_accuracy.png
│   ├── cnn_confusion_matrix.png
│   ├── dna_splice_cnn.keras
│   ├── logistic_confusion_matrix.png
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── random_forest_confusion_matrix.png
│   └── results.json
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

Dataset

The project uses the UCI Molecular Biology Splice Junction Gene Sequences dataset.

Dataset characteristics:

* 3,190 DNA sequences
* 60 nucleotide positions per sequence
* 3 classification classes
* Classes: EI, IE, and N

Source:

UCI Machine Learning Repository — Molecular Biology (Splice-junction Gene Sequences)

The dataset is used for educational and machine-learning research purposes.

Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* UCI ML Repository

Project Workflow

UCI Dataset
     ↓
Data Loading
     ↓
DNA Sequence Construction
     ↓
Data Cleaning
     ↓
Train/Test Split
     ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
3-mer Features       One-Hot Encoding
 │                       │
 ▼                       ▼
Logistic Regression   1D CNN
Random Forest            │
 │                       │
 └───────────┬───────────┘
             ▼
       Model Evaluation
             ↓
     Confusion Matrices
             ↓
       Results Analysis
             ↓
      Streamlit Application

Future Improvements

Potential future extensions include:

* Macro-averaged evaluation metrics
* ROC-AUC analysis
* Hyperparameter optimization
* More robust validation strategies
* Additional sequence representations
* Data-leakage analysis based on sequence similarity
* Comparison with additional deep-learning architectures
* Improved biological interpretation of learned sequence patterns

Disclaimer

This project is an educational machine-learning project and is not intended for medical or clinical use.
