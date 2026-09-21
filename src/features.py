import itertools
from collections import Counter

import numpy as np


BASE_TO_VECTOR = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1]
}


def one_hot_encode_sequence(sequence):
    """Convert one DNA sequence into a one-hot encoded array."""

    encoded = []

    for base in sequence:
        encoded.append(
            BASE_TO_VECTOR.get(
                base,
                [0, 0, 0, 0]
            )
        )

    return np.array(
        encoded,
        dtype=np.float32
    )


def one_hot_encode_sequences(sequences):
    """One-hot encode multiple DNA sequences."""

    return np.array([
        one_hot_encode_sequence(sequence)
        for sequence in sequences
    ])


def kmer_features(sequences, k=3):
    """Create normalized k-mer frequency features."""

    all_kmers = [
        "".join(chars)
        for chars in itertools.product(
            "ACGT",
            repeat=k
        )
    ]

    features = []

    for sequence in sequences:

        kmers = [
            sequence[i:i + k]
            for i in range(len(sequence) - k + 1)
        ]

        counts = Counter(kmers)

        total = max(len(kmers), 1)

        row = [
            counts[kmer] / total
            for kmer in all_kmers
        ]

        features.append(row)

    return np.array(
        features,
        dtype=np.float32
    )