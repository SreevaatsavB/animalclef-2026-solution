"""
Data loading utilities for AnimalCLEF 2026
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset


class AnimalCLEFDataset(Dataset):
    """PyTorch Dataset for AnimalCLEF 2026 competition data"""

    def __init__(self, metadata_path: str, data_dir: str, transform=None):
        """
        Args:
            metadata_path: Path to metadata.csv
            data_dir: Base data directory (e.g., 'data/raw')
            transform: Optional transform to apply to images
        """
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: Transformed image tensor
            info: Dictionary with metadata (identity, dataset, etc.)
        """
        row = self.metadata.iloc[idx]

        # Construct image path (path column already includes 'images/')
        img_path = self.data_dir / row['path']

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Metadata info
        info = {
            'identity': row.get('identity', None),
            'dataset': row.get('dataset', None),
            'image_path': str(img_path)
        }

        return image, info


def load_wildlife_dataset(dataset_name: str, data_dir: str):
    """
    Load a dataset from WildlifeDatasets package

    Args:
        dataset_name: Name of the dataset (e.g., 'MacaqueFaces')
        data_dir: Directory to store/load the dataset

    Returns:
        dataset: Loaded wildlife dataset
    """
    from wildlife_datasets import datasets

    # Get dataset class
    dataset_class = getattr(datasets, dataset_name)

    # Download if needed
    dataset_path = Path(data_dir) / dataset_name
    if not dataset_path.exists():
        print(f"Downloading {dataset_name}...")
        dataset_class.get_data(str(dataset_path))

    # Load dataset
    dataset = dataset_class(str(dataset_path))
    return dataset


def get_available_wildlife_datasets() -> List[str]:
    """Get list of available wildlife datasets"""
    from wildlife_datasets import datasets

    available = [
        name for name in dir(datasets)
        if not name.startswith('_') and name[0].isupper()
    ]
    return available
