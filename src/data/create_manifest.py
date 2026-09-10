from pathlib import Path
import pandas as pd


DATASET_ROOT = Path("data/raw/cifake")
OUTPUT_FILE = Path("data/processed/cifake_manifest.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(split, label_name, label):
    folder = DATASET_ROOT / split / label_name

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows = []

    for image_path in folder.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append(
                {
                    "path": image_path.as_posix(),
                    "label": label,
                    "generator": (
                        "stable_diffusion_v1_4"
                        if label == 1
                        else "real_cifar10"
                    ),
                    "source": "CIFAKE",
                    "split": split,
                }
            )

    return rows


def main():
    rows = []

    rows.extend(collect_images("train", "REAL", 0))
    rows.extend(collect_images("train", "FAKE", 1))
    rows.extend(collect_images("test", "REAL", 0))
    rows.extend(collect_images("test", "FAKE", 1))

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created manifest: {OUTPUT_FILE}")
    print(f"Total images: {len(df)}")
    print("\nImages by split:")
    print(df["split"].value_counts())

    print("\nImages by label:")
    print(df["label"].value_counts())

    print("\nImages by split and label:")
    print(df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()