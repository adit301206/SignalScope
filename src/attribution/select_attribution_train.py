from pathlib import Path
import csv
import io
import random

import pyarrow.parquet as pq
from PIL import Image


# ==============================================================
# Configuration
# ==============================================================

DATA_DIR = Path(
    "data/raw/defactify_full/data"
)

OUTPUT_DIR = Path(
    "data/raw/defactify_attribution"
)

IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

SEED = 42

# 5,000 images per class
TARGETS = {
    0: 5000,  # Real
    1: 5000,  # Stable Diffusion 2.1
    2: 5000,  # SDXL
    3: 5000,  # Stable Diffusion 3
    4: 5000,  # DALL-E 3
    5: 5000,  # Midjourney 6
}

GENERATOR_NAMES = {
    0: "real",
    1: "stable_diffusion_2_1",
    2: "stable_diffusion_xl",
    3: "stable_diffusion_3",
    4: "dalle_3",
    5: "midjourney_6",
}


def main():

    random.seed(SEED)

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_files = sorted(
        DATA_DIR.glob("train-*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No training Parquet files found in {DATA_DIR}"
        )

    print("=" * 65)
    print("SignalScope - Attribution Dataset Selection")
    print("=" * 65)

    print(f"Found {len(parquet_files)} Parquet files.")

    print("\nTarget per class:")
    for label, target in TARGETS.items():
        print(
            f"  {label}: "
            f"{GENERATOR_NAMES[label]} = {target}"
        )

    print("\nTotal target: 30,000 images")
    print("=" * 65)

    # ----------------------------------------------------------
    # Reservoirs
    # ----------------------------------------------------------

    reservoirs = {
        label: []
        for label in TARGETS
    }

    seen = {
        label: 0
        for label in TARGETS
    }

    global_index = 0

    # ----------------------------------------------------------
    # Scan Parquet files
    # ----------------------------------------------------------

    for file_number, parquet_path in enumerate(
        parquet_files,
        start=1,
    ):

        print(
            f"\n[{file_number}/{len(parquet_files)}] "
            f"Reading {parquet_path.name}..."
        )

        parquet_file = pq.ParquetFile(
            parquet_path
        )

        for batch in parquet_file.iter_batches(
            columns=["Image", "Label_B"],
            batch_size=512,
        ):

            rows = batch.to_pylist()

            for row in rows:

                label = row["Label_B"]

                if label not in TARGETS:
                    global_index += 1
                    continue

                image_data = row["Image"]

                if image_data is None:
                    global_index += 1
                    continue

                image_bytes = image_data.get("bytes")

                if not image_bytes:
                    global_index += 1
                    continue

                seen[label] += 1

                item = {
                    "global_index": global_index,
                    "label_b": label,
                    "image_bytes": image_bytes,
                }

                reservoir = reservoirs[label]
                target = TARGETS[label]

                if len(reservoir) < target:

                    reservoir.append(item)

                else:

                    replacement = random.randint(
                        0,
                        seen[label] - 1,
                    )

                    if replacement < target:
                        reservoir[replacement] = item

                global_index += 1

        print(
            "  Selected: "
            + ", ".join(
                f"{GENERATOR_NAMES[label]}="
                f"{len(reservoirs[label])}/{TARGETS[label]}"
                for label in TARGETS
            )
        )

    # ----------------------------------------------------------
    # Verify selection
    # ----------------------------------------------------------

    print("\nFinished scanning all Parquet files.")

    for label, target in TARGETS.items():

        if len(reservoirs[label]) < target:

            raise RuntimeError(
                f"Not enough samples for "
                f"{GENERATOR_NAMES[label]}: "
                f"{len(reservoirs[label])}/{target}"
            )

    # ----------------------------------------------------------
    # Save images
    # ----------------------------------------------------------

    print("\nSaving selected images...")

    manifest_rows = []

    for label in TARGETS:

        generator = GENERATOR_NAMES[label]

        print(
            f"  Saving {target if (target := TARGETS[label]) else 0} "
            f"{generator} images..."
        )

        for item in reservoirs[label]:

            try:

                image = Image.open(
                    io.BytesIO(
                        item["image_bytes"]
                    )
                ).convert("RGB")

            except Exception as exc:

                print(
                    f"WARNING: Could not decode image "
                    f"{item['global_index']}: {exc}"
                )

                continue

            filename = (
                f"{generator}_"
                f"{item['global_index']:06d}.jpg"
            )

            output_path = (
                IMAGE_DIR / filename
            )

            image.save(
                output_path,
                "JPEG",
                quality=95,
            )

            manifest_rows.append(
                {
                    "filename": filename,
                    "label": 0 if label == 0 else 1,
                    "generator": generator,
                    "label_b": label,
                    "source": "Defactify",
                    "split": "train",
                    "original_index": item[
                        "global_index"
                    ],
                }
            )

    # ----------------------------------------------------------
    # Write manifest
    # ----------------------------------------------------------

    with open(
        MANIFEST_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "label",
                "generator",
                "label_b",
                "source",
                "split",
                "original_index",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n" + "=" * 65)
    print("ATTRIBUTION DATASET CREATED")
    print("=" * 65)

    print(
        f"Total images saved: {len(manifest_rows):,}"
    )

    print(
        f"Image directory: {IMAGE_DIR}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print("\nFinal counts:")

    for label, name in GENERATOR_NAMES.items():

        count = sum(
            1
            for row in manifest_rows
            if row["generator"] == name
        )

        print(
            f"  {name}: {count:,}"
        )

    print("=" * 65)


if __name__ == "__main__":
    main()