from pathlib import Path
import json

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import (
    load_raw_data,
    build_sequences,
    clean_sequences,
    prepare_labels,
    split_data
)

from features import (
    kmer_features,
    one_hot_encode_sequences
)


RANDOM_STATE = 42


def evaluate_model(name, y_true, y_pred):
    """Calculate and print classification metrics."""

    results = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred
        ).tolist()
    }

    print(f"\n{'=' * 50}")
    print(name)
    print(f"{'=' * 50}")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["EI", "IE", "N"],
            zero_division=0
        )
    )

    print("Accuracy:", results["accuracy"])
    print("Weighted F1:", results["f1_weighted"])

    return results


def plot_confusion_matrix(
    matrix,
    title,
    output_path
):
    """Save a confusion matrix visualization."""

    plt.figure(figsize=(6, 5))

    plt.imshow(matrix)

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    class_names = ["EI", "IE", "N"]

    plt.xticks(
        range(len(class_names)),
        class_names
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            plt.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


def build_cnn(input_shape, num_classes=3):
    """Build the 1D CNN model."""

    model = Sequential([
        Conv1D(
            64,
            kernel_size=5,
            activation="relu",
            input_shape=input_shape
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            128,
            kernel_size=3,
            activation="relu"
        ),

        GlobalMaxPooling1D(),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    project_root = Path(__file__).resolve().parents[1]

    models_dir = project_root / "models"

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading dataset...")

    X, y = load_raw_data()

    sequences = clean_sequences(
        build_sequences(X)
    )

    labels, _ = prepare_labels(y)

    print(
        f"Total sequences: {len(sequences)}"
    )

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------

    sequences_train, sequences_test, y_train, y_test = split_data(
        sequences,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------
    # Classical Machine Learning
    # --------------------------------------------------

    print(
        "\nCreating k-mer features..."
    )

    X_train_kmer = kmer_features(
        sequences_train,
        k=3
    )

    X_test_kmer = kmer_features(
        sequences_test,
        k=3
    )

    print(
        "K-mer feature shape:",
        X_train_kmer.shape
    )

    results = {}

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    logistic_model.fit(
        X_train_kmer,
        y_train
    )

    logistic_predictions = logistic_model.predict(
        X_test_kmer
    )

    results["logistic_regression"] = evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_predictions
    )

    joblib.dump(
        logistic_model,
        models_dir / "logistic_regression.joblib"
    )

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    random_forest.fit(
        X_train_kmer,
        y_train
    )

    random_forest_predictions = random_forest.predict(
        X_test_kmer
    )

    results["random_forest"] = evaluate_model(
        "Random Forest",
        y_test,
        random_forest_predictions
    )

    joblib.dump(
        random_forest,
        models_dir / "random_forest.joblib"
    )

    # --------------------------------------------------
    # Deep Learning
    # --------------------------------------------------

    print(
        "\nPreparing DNA sequences for CNN..."
    )

    X_train_cnn = one_hot_encode_sequences(
        sequences_train
    )

    X_test_cnn = one_hot_encode_sequences(
        sequences_test
    )

    print(
        "CNN input shape:",
        X_train_cnn.shape
    )

    print(
        "\nBuilding CNN..."
    )

    cnn = build_cnn(
        input_shape=X_train_cnn.shape[1:],
        num_classes=3
    )

    cnn.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    print(
        "\nTraining CNN..."
    )

    history = cnn.fit(
        X_train_cnn,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # --------------------------------------------------
    # CNN Evaluation
    # --------------------------------------------------

    cnn_probabilities = cnn.predict(
        X_test_cnn,
        verbose=0
    )

    cnn_predictions = np.argmax(
        cnn_probabilities,
        axis=1
    )

    results["cnn"] = evaluate_model(
        "1D CNN",
        y_test,
        cnn_predictions
    )

    # --------------------------------------------------
    # CNN Training Accuracy Plot
    # --------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "CNN Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        models_dir / "cnn_accuracy.png"
    )

    plt.close()

    # --------------------------------------------------
    # Save CNN
    # --------------------------------------------------

    cnn.save(
        models_dir / "dna_splice_cnn.keras"
    )

    # --------------------------------------------------
    # Save Confusion Matrices
    # --------------------------------------------------

    plot_confusion_matrix(
        results["logistic_regression"]["confusion_matrix"],
        "Logistic Regression Confusion Matrix",
        models_dir / "logistic_confusion_matrix.png"
    )

    plot_confusion_matrix(
        results["random_forest"]["confusion_matrix"],
        "Random Forest Confusion Matrix",
        models_dir / "random_forest_confusion_matrix.png"
    )

    plot_confusion_matrix(
        results["cnn"]["confusion_matrix"],
        "1D CNN Confusion Matrix",
        models_dir / "cnn_confusion_matrix.png"
    )

    # --------------------------------------------------
    # Save Results
    # --------------------------------------------------

    results_path = models_dir / "results.json"

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "\nTraining complete!"
    )

    print(
        f"Models and results saved in: {models_dir}"
    )


if __name__ == "__main__":
    main()