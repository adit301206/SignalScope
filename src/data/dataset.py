from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SignalScopeDataset(Dataset):
    """
    Dataset for SignalScope image classification.

    Expected CSV columns:
        path      -> image file path
        label     -> 0 for real, 1 for AI-generated
        generator -> generator/source name
        split     -> train / val / test
    """

    def __init__(self, csv_file, transform=None):
        self.csv_file = Path(csv_file)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_file}"
            )

        self.data = pd.read_csv(self.csv_file)

        required_columns = {"path", "label"}

        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = Path(row["path"])
        label = int(row["label"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label