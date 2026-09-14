from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

from src.attribution.model import (
    GENERATOR_CLASSES,
    create_attribution_model,
)
from src.data.transforms import get_eval_transforms


# ==============================================================
# Configuration
# ==============================================================

MANIFEST_PATH = Path(
    "data/raw/defactify_eval/manifest.csv"
)

CHECKPOINT_PATH = Path(
    "model/generator_attribution_30k.pth"
)

OUTPUT_DIR = Path(
    "reports/attribution"
)

BATCH_SIZE = 16
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================================================
# Dataset
# ==============================================================

class AttributionEvalDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # The evaluation manifest already contains
        # the complete image path.
        image_path = Path(str(row["path"]))

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # IMPORTANT:
        # The evaluation manifest uses:
        # generator_id = 0..5
        #
        # label is only the binary Real/AI label.
        label = int(row["generator_id"])

        return image, label


# ==============================================================
# Main
# ==============================================================

def main():

    print("\n" + "=" * 70)
    print("SignalScope - 30K Generator Attribution Evaluation")
    print("=" * 70)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ----------------------------------------------------------
    # Check files
    # ----------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation manifest not found: {MANIFEST_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # ----------------------------------------------------------
    # Load manifest
    # ----------------------------------------------------------

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        "\nEvaluation samples:",
        len(df)
    )

    print(
        "\nEvaluation class distribution:"
    )

    print(
        df["generator"]
        .value_counts()
        .sort_index()
    )

    # ----------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------

    assert len(df) == 600, (
        f"Expected 600 evaluation images, "
        f"found {len(df)}"
    )

    assert df["generator_id"].nunique() == 6, (
        "Expected exactly 6 attribution classes."
    )

    expected_ids = set(range(6))

    actual_ids = set(
        df["generator_id"].unique()
    )

    assert actual_ids == expected_ids, (
        f"Expected generator IDs {expected_ids}, "
        f"found {actual_ids}"
    )

    print(
        "\nLabel distribution:"
    )

    for label in sorted(
        df["generator_id"].unique()
    ):

        count = (
            df["generator_id"] == label
        ).sum()

        print(
            f"  {label}: "
            f"{GENERATOR_CLASSES[int(label)]} "
            f"-> {count}"
        )

    # ----------------------------------------------------------
    # Verify image files
    # ----------------------------------------------------------

    missing_images = sum(
        not Path(str(path)).exists()
        for path in df["path"]
    )

    if missing_images > 0:
        raise FileNotFoundError(
            f"{missing_images} evaluation images are missing."
        )

    print(
        "\nAll evaluation images found."
    )

    # ----------------------------------------------------------
    # Dataset / DataLoader
    # ----------------------------------------------------------

    dataset = AttributionEvalDataset(
        df,
        transform=get_eval_transforms()
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    print(
        "\nLoading 30k attribution model..."
    )

    model = create_attribution_model(
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        "Checkpoint validation Macro-F1:",
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    labels = np.array(
        all_labels
    )

    predictions = np.array(
        all_predictions
    )

    probabilities = np.array(
        all_probabilities
    )

    # ----------------------------------------------------------
    # Overall metrics
    # ----------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:  {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro-F1:  {macro_f1:.4f}"
    )

    # ----------------------------------------------------------
    # Confusion matrix
    # ----------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(
            range(len(GENERATOR_CLASSES))
        )
    )

    cm_df = pd.DataFrame(
        cm,
        index=GENERATOR_CLASSES,
        columns=GENERATOR_CLASSES
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        cm_df.to_string()
    )

    # ----------------------------------------------------------
    # Classification report
    # ----------------------------------------------------------

    report_text = classification_report(
        labels,
        predictions,
        labels=list(
            range(len(GENERATOR_CLASSES))
        ),
        target_names=GENERATOR_CLASSES,
        zero_division=0
    )

    report_dict = classification_report(
        labels,
        predictions,
        labels=list(
            range(len(GENERATOR_CLASSES))
        ),
        target_names=GENERATOR_CLASSES,
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    print(
        report_text
    )

    # ----------------------------------------------------------
    # Top-2 accuracy
    # ----------------------------------------------------------

    top2_predictions = np.argsort(
        probabilities,
        axis=1
    )[:, -2:]

    top2_correct = np.array([
        label in top2_predictions[i]
        for i, label in enumerate(labels)
    ])

    top2_accuracy = top2_correct.mean()

    print(
        f"Top-2 Accuracy: "
        f"{top2_accuracy:.4f} "
        f"({top2_accuracy * 100:.2f}%)"
    )

    # ----------------------------------------------------------
    # Generator-wise accuracy
    # ----------------------------------------------------------

    per_generator = []

    print("\n" + "=" * 70)
    print("GENERATOR-WISE ACCURACY")
    print("=" * 70)

    for class_id, class_name in enumerate(
        GENERATOR_CLASSES
    ):

        mask = labels == class_id

        class_accuracy = accuracy_score(
            labels[mask],
            predictions[mask]
        )

        class_f1 = report_dict[
            class_name
        ]["f1-score"]

        correct = int(
            (predictions[mask] == class_id).sum()
        )

        total = int(
            mask.sum()
        )

        print(
            f"{class_name:25s} "
            f"{class_accuracy * 100:6.2f}% "
            f"({correct}/{total})"
        )

        per_generator.append(
            {
                "generator": class_name,
                "samples": total,
                "correct": correct,
                "accuracy": class_accuracy,
                "f1": class_f1,
            }
        )

    per_generator_df = pd.DataFrame(
        per_generator
    )

    # ----------------------------------------------------------
    # Save reports
    # ----------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cm_df.to_csv(
        OUTPUT_DIR /
        "confusion_matrix_30k.csv"
    )

    per_generator_df.to_csv(
        OUTPUT_DIR /
        "generator_results_30k.csv",
        index=False
    )

    report_path = (
        OUTPUT_DIR /
        "evaluation_30k.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SignalScope - 30K Generator Attribution Evaluation\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Checkpoint: "
            f"{CHECKPOINT_PATH}\n"
        )

        f.write(
            f"Evaluation samples: "
            f"{len(df)}\n"
        )

        f.write(
            f"Validation Accuracy: "
            f"{checkpoint['val_accuracy']:.4f}\n"
        )

        f.write(
            f"Validation Macro-F1: "
            f"{checkpoint['val_macro_f1']:.4f}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{accuracy:.4f}\n"
        )

        f.write(
            f"Test Macro-F1: "
            f"{macro_f1:.4f}\n"
        )

        f.write(
            f"Top-2 Accuracy: "
            f"{top2_accuracy:.4f}\n\n"
        )

        f.write(
            "Confusion Matrix:\n\n"
        )

        f.write(
            cm_df.to_string()
        )

        f.write(
            "\n\nClassification Report:\n\n"
        )

        f.write(
            report_text
        )

        f.write(
            "\nGenerator-wise Accuracy:\n\n"
        )

        for row in per_generator:
            f.write(
                f"{row['generator']}: "
                f"{row['accuracy']:.4f} "
                f"({row['correct']}/{row['samples']})\n"
            )

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro-F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Top-2 Accuracy: "
        f"{top2_accuracy * 100:.2f}%"
    )

    print(
        "\nReports saved to:",
        OUTPUT_DIR
    )

    print(
        "\nFiles:"
    )

    print(
        "  - evaluation_30k.txt"
    )

    print(
        "  - confusion_matrix_30k.csv"
    )

    print(
        "  - generator_results_30k.csv"
    )


if __name__ == "__main__":
    main()