from pathlib import Path
from collections import Counter
from urllib.parse import urlparse
import csv
import time

import requests


API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "Rajarshi-Roy-research/Defactify_Image_Dataset"
CONFIG = "default"
SPLIT = "test"

# We will first test with 10 images per category.
# After it works, change this to 100.
TARGET_PER_LABEL = 100

PAGE_SIZE = 100
MAX_SCAN_ROWS = 2000

OUTPUT_DIR = Path("data/raw/defactify_eval")
IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

GENERATOR_NAMES = {
    0: "real",
    1: "stable_diffusion_2_1",
    2: "sdxl",
    3: "stable_diffusion_3",
    4: "dalle3",
    5: "midjourney_6",
}


def get_json(session, params):
    """Get JSON from the Dataset Viewer API with retries."""
    for attempt in range(5):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()

        except Exception as error:
            if attempt == 4:
                raise

            wait_time = 2 ** attempt
            print(
                f"Request failed: {error}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)


def download_image(session, url, output_path):
    """Download one image with retries."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    for attempt in range(5):
        try:
            response = session.get(
                url,
                timeout=60,
            )
            response.raise_for_status()

            output_path.write_bytes(response.content)
            return

        except Exception as error:
            if attempt == 4:
                raise

            wait_time = 2 ** attempt
            print(
                f"Image download failed: {error}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # Store selected rows for each generator/category.
    selected = {
        label: []
        for label in GENERATOR_NAMES
    }

    print("Scanning Defactify test split...")
    print(
        f"Target: {TARGET_PER_LABEL} images "
        f"per category"
    )

    # ---------------------------------------------------------
    # STEP 1: Find a balanced set of rows
    # ---------------------------------------------------------

    for offset in range(0, MAX_SCAN_ROWS, PAGE_SIZE):

        params = {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": PAGE_SIZE,
        }

        data = get_json(session, params)

        rows = data.get("rows", [])

        if not rows:
            print("No more rows returned.")
            break

        for item in rows:
            row = item["row"]

            generator_id = int(row["Label_B"])

            if generator_id not in selected:
                continue

            if len(selected[generator_id]) < TARGET_PER_LABEL:
                selected[generator_id].append(
                    {
                        "row_idx": item.get("row_idx"),
                        "row": row,
                    }
                )

        counts = {
            GENERATOR_NAMES[label]: len(items)
            for label, items in selected.items()
        }

        print(
            f"Scanned rows 0-{offset + len(rows)} | "
            f"{counts}"
        )

        if all(
            len(items) >= TARGET_PER_LABEL
            for items in selected.values()
        ):
            break

    # ---------------------------------------------------------
    # STEP 2: Verify we found enough images
    # ---------------------------------------------------------

    missing = {
        GENERATOR_NAMES[label]: len(items)
        for label, items in selected.items()
        if len(items) < TARGET_PER_LABEL
    }

    if missing:
        raise RuntimeError(
            "Could not find enough images for: "
            f"{missing}"
        )

    print("\nBalanced sample found:")
    for label, items in selected.items():
        print(
            f"  {GENERATOR_NAMES[label]}: "
            f"{len(items)}"
        )

    # ---------------------------------------------------------
    # STEP 3: Download images
    # ---------------------------------------------------------

    manifest_rows = []

    total = TARGET_PER_LABEL * len(GENERATOR_NAMES)
    downloaded = 0

    print(f"\nDownloading {total} images...")

    for generator_id, items in selected.items():

        generator_name = GENERATOR_NAMES[generator_id]

        for number, item in enumerate(items, start=1):

            row = item["row"]
            row_idx = item["row_idx"]

            image_info = row["Image"]

            if not isinstance(image_info, dict):
                raise RuntimeError(
                    f"Unexpected Image field for row {row_idx}"
                )

            image_url = image_info.get("src")

            if not image_url:
                raise RuntimeError(
                    f"No image URL found for row {row_idx}"
                )

            # Preserve the original image extension when possible.
            extension = Path(
                urlparse(image_url).path
            ).suffix.lower()

            if extension not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:
                extension = ".jpg"

            filename = (
                f"{generator_name}_{number:03d}"
                f"{extension}"
            )

            output_path = IMAGE_DIR / filename

            download_image(
                session,
                image_url,
                output_path,
            )

            downloaded += 1

            print(
                f"[{downloaded}/{total}] "
                f"{filename}"
            )

            # Save metadata for our evaluation pipeline.
            manifest_rows.append(
                {
                    "path": output_path.as_posix(),
                    "label": 0 if generator_id == 0 else 1,
                    "generator": generator_name,
                    "generator_id": generator_id,
                    "source": "Defactify_Image_Dataset",
                    "split": "test",
                    "row_idx": row_idx,
                }
            )

    # ---------------------------------------------------------
    # STEP 4: Save manifest
    # ---------------------------------------------------------

    fieldnames = [
        "path",
        "label",
        "generator",
        "generator_id",
        "source",
        "split",
        "row_idx",
    ]

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    # ---------------------------------------------------------
    # STEP 5: Final summary
    # ---------------------------------------------------------

    print("\n========================================")
    print("Defactify evaluation sample ready!")
    print("========================================")

    print(f"Images:   {IMAGE_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")

    print("\nGenerator counts:")
    print(
        Counter(
            row["generator"]
            for row in manifest_rows
        )
    )

    print("\nLabel counts:")
    print(
        Counter(
            row["label"]
            for row in manifest_rows
        )
    )


if __name__ == "__main__":
    main()