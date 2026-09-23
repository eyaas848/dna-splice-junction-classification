from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from src.features import one_hot_encode_sequence


CLASS_NAMES = {
    0: "EI",
    1: "IE",
    2: "Neither"
}


def clean_sequence(sequence):
    """Clean and validate a DNA sequence."""

    sequence = sequence.strip().upper()

    valid_bases = set("ACGTN")

    cleaned = ""

    for base in sequence:
        if base in valid_bases:
            cleaned += base
        else:
            raise ValueError(
                f"Invalid DNA base detected: {base}"
            )

    return cleaned


def load_cnn_model():
    """Load the trained CNN model."""

    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "models"
        / "dna_splice_cnn.keras"
    )

    return load_model(model_path)


def predict_sequence(sequence):
    """Predict the splice-junction class of a DNA sequence."""

    sequence = clean_sequence(sequence)

    if not sequence:
        raise ValueError(
            "Please enter a DNA sequence."
        )
    
    if len(sequence) != 60:
        raise ValueError(
            f"Please enter exactly 60 DNA bases. "
            f"Your sequence contains {len(sequence)} bases."
        )

    model = load_cnn_model()

    encoded_sequence = one_hot_encode_sequence(
        sequence
    )

    encoded_sequence = np.expand_dims(
        encoded_sequence,
        axis=0
    )

    probabilities = model.predict(
        encoded_sequence,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    return {
        "class": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }
    }


if __name__ == "__main__":
    example_sequence = (
        "ATGCGTACGTTAGCGATCGATCGATCGATCGATCGATCG"
    )

    result = predict_sequence(
        example_sequence
    )

    print("Prediction:", result["class"])
    print(
        "Confidence:",
        f"{result['confidence']:.2%}"
    )

    print("Probabilities:")

    for label, probability in result["probabilities"].items():
        print(
            f"{label}: {probability:.2%}"
        )