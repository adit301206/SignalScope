from pathlib import Path
import csv
import time
import requests
from PIL import Image
from io import BytesIO


DATASET = "Rajarshi-Roy-research/Defactify_Image_Dataset"
CONFIG = "default"
SPLIT = "train"

# Balanced binary training sample:
# 500 real + 500 AI-generated
TARGET_REAL = 500
TARGET_PER_GENERATOR = 100

PAGE_SIZE = 100

OUTPUT_DIR = Path("data/raw/defactify_train")
IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

API_URL = "https://datasets-server.huggingface.co/rows"

GENERATOR_NAMES = {
    0: "real",
    1: "stable_diffusion_2_1",
    2: "stable_diffusion_xl",
    3: "stable_diffusion_3",
    4: "dalle_3",
    5: "midjourney_6",
}


def get_rows(session, offset):
    """Get one page of 100 rows from the Defactify training split."""

    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": PAGE_SIZE,
    }

    for attempt in range(1, 6):
        try:
            print(
                f"Requesting rows {offset}–"
                f"{offset + PAGE_SIZE - 1} "
                f"(attempt {attempt}/5)..."
            )

            response = session.get(
                API_URL,
                params=params,
                timeout=(15, 180),
            )

            if response.status_code == 429:
                wait_time = 20 * attempt
                print(
                    f"Rate limited (429). "
                    f"Waiting {wait_time} seconds..."
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            wait_time = 10 * attempt
            print(
                f"Request timed out. "
                f"Waiting {wait_time} seconds..."
            )
            time.sleep(wait_time)

        except requests.exceptions.RequestException as exc:
            print(f"Request error: {exc}")

            if attempt < 5:
                time.sleep(10 * attempt)
            else:
                raise

    raise RuntimeError(
        "Could not retrieve rows from Hugging Face "
        "after 5 attempts."
    )


def download_image(session, url, output_path):
    """Download and save one image."""

    if output_path.exists():
        return True

    for attempt in range(1, 4):
        try:
            response = session.get(
                url,
                timeout=(15, 120),
            )
            response.raise_for_status()

            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

            image.save(
                output_path,
                "JPEG",
                quality=95,
            )

            return True

        except Exception as exc:
            print(
                f"Image download failed "
                f"(attempt {attempt}/3): {exc}"
            )
            time.sleep(3)

    return False


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    selected = []

    counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }

    targets = {
        0: TARGET_REAL,
        1: TARGET_PER_GENERATOR,
        2: TARGET_PER_GENERATOR,
        3: TARGET_PER_GENERATOR,
        4: TARGET_PER_GENERATOR,
        5: TARGET_PER_GENERATOR,
    }

    print("Starting Defactify training-data selection...")
    print("Target: 500 real + 500 AI = 1000 images")
    print()

    # We scan the training split in small pages.
    # We stop immediately once all targets are reached.
    offset = 0

    while True:
        data = get_rows(session, offset)

        rows = data.get("rows", [])

        if not rows:
            print("No more rows returned.")
            break

        for item in rows:
            row = item.get("row", {})

            label_b = row.get("Label_B")

            # Ignore unexpected labels.
            if label_b not in targets:
                continue

            # Skip this category if its target is already full.
            if counts[label_b] >= targets[label_b]:
                continue

            image_data = row.get("image")

            if not image_data:
                continue

            image_url = image_data.get("src")

            if not image_url:
                continue

            selected.append({
                "row_idx": item["row_idx"],
                "label": 0 if label_b == 0 else 1,
                "generator": GENERATOR_NAMES[label_b],
                "url": image_url,
            })

            counts[label_b] += 1

        print(
            "Progress: "
            f"real={counts[0]}/500, "
            f"SD2.1={counts[1]}/100, "
            f"SDXL={counts[2]}/100, "
            f"SD3={counts[3]}/100, "
            f"DALL-E3={counts[4]}/100, "
            f"Midjourney={counts[5]}/100"
        )

        complete = all(
            counts[label] >= targets[label]
            for label in targets
        )

        if complete:
            print()
            print("All 1000 required images found.")
            break

        offset += PAGE_SIZE

        # Safety limit.
        if offset >= 42000:
            print("Reached the end of the training split.")
            break

        time.sleep(2)

    print()
    print("Selection finished.")
    print(f"Selected: {len(selected)} images")
    print()

    if len(selected) != 1000:
        raise RuntimeError(
            f"Expected 1000 images, "
            f"but selected {len(selected)}."
        )

    # ---------------------------------------------------------
    # Download selected images
    # ---------------------------------------------------------

    manifest_rows = []

    print("Downloading selected images...")
    print()

    for index, item in enumerate(
        selected,
        start=1,
    ):
        filename = (
            f"{item['generator']}_"
            f"{item['row_idx']}_"
            f"{index:04d}.jpg"
        )

        output_path = IMAGE_DIR / filename

        success = download_image(
            session,
            item["url"],
            output_path,
        )

        if not success:
            print(
                f"Skipping failed image: {filename}"
            )
            continue

        manifest_rows.append({
            "filename": filename,
            "label": item["label"],
            "generator": item["generator"],
            "source": DATASET,
            "split": SPLIT,
            "row_idx": item["row_idx"],
        })

        if index % 25 == 0:
            print(
                f"Downloaded {index}/1000"
            )

        time.sleep(0.1)

    # ---------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------

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
                "source",
                "split",
                "row_idx",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=" * 50)
    print("FINISHED")
    print("=" * 50)
    print(f"Images downloaded: {len(manifest_rows)}")
    print(f"Manifest: {MANIFEST_PATH}")
    print()
    print("Generator counts:")

    for label, name in GENERATOR_NAMES.items():
        amount = sum(
            1
            for row in manifest_rows
            if row["generator"] == name
        )
        print(f"  {name}: {amount}")


if __name__ == "__main__":
    main()