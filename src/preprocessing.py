from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data():
    """Load the downloaded feature and target files."""

    project_root = Path(__file__).resolve().parents[1]

    features_path = project_root / "data" / "raw" / "features.csv"
    targets_path = project_root / "data" / "raw" / "targets.csv"

    X = pd.read_csv(features_path)
    y = pd.read_csv(targets_path)

    return X, y


def build_sequences(X):
    """Combine the nucleotide-position columns into DNA sequences."""

    sequences = X.astype(str).apply(
        lambda row: "".join(row),
        axis=1
    )

    return sequences


def clean_sequences(sequences):
    """Clean DNA sequences and replace unexpected characters with N."""

    valid_bases = set("ACGTN")

    def clean(sequence):
        sequence = sequence.strip().upper()

        return "".join(
            base if base in valid_bases else "N"
            for base in sequence
        )

    return sequences.apply(clean)


def prepare_labels(y):
    """Convert class labels into numerical values."""

    labels = y.iloc[:, 0].astype(str).str.strip().str.upper()

    label_mapping = {
        "EI": 0,
        "IE": 1,
        "N": 2
    }

    encoded = labels.map(label_mapping)

    if encoded.isna().any():
        unknown_labels = labels[encoded.isna()].unique()
        raise ValueError(
            f"Unknown class label(s) detected: {unknown_labels}"
        )

    return encoded, labels


def split_data(
    sequences,
    labels,
    test_size=0.2,
    random_state=42
):
    """Split sequences and labels into training and testing sets."""

    return train_test_split(
        sequences,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )


if __name__ == "__main__":
    X, y = load_raw_data()

    sequences = clean_sequences(
        build_sequences(X)
    )

    labels, original_labels = prepare_labels(y)

    X_train, X_test, y_train, y_test = split_data(
        sequences,
        labels
    )

    print("Dataset prepared.")
    print(f"Total samples: {len(sequences)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nClass distribution:")
    print(original_labels.value_counts())