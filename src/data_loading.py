import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class AnimalCLEFDataset(Dataset):
    def __init__(self, metadata_path, data_dir, transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = self.data_dir / row['path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        info = {
            'identity': row.get('identity', None),
            'dataset': row.get('dataset', None),
            'image_path': str(img_path)
        }
        return image, info


def load_wildlife_dataset(dataset_name, data_dir):
    from wildlife_datasets import datasets

    dataset_class = getattr(datasets, dataset_name)
    dataset_path = Path(data_dir) / dataset_name

    if not dataset_path.exists():
        print(f"Downloading {dataset_name}...")
        dataset_class.get_data(str(dataset_path))

    return dataset_class(str(dataset_path))


def get_available_wildlife_datasets():
    from wildlife_datasets import datasets
    return [name for name in dir(datasets) if not name.startswith('_') and name[0].isupper()]
