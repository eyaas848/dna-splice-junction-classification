from pathlib import Path
from ucimlrepo import fetch_ucirepo


def download_dataset():
    """Download the UCI Molecular Biology Splice Junction dataset."""

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI Splice Junction dataset...")

    dataset = fetch_ucirepo(id=69)

    features = dataset.data.features
    targets = dataset.data.targets

    features.to_csv(raw_dir / "features.csv", index=False)
    targets.to_csv(raw_dir / "targets.csv", index=False)

    print("Dataset downloaded successfully.")
    print(f"Samples: {len(features)}")


if __name__ == "__main__":
    download_dataset()