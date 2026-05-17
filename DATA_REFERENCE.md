# Data Reference - AnimalCLEF 2026

## Metadata Columns

The `data/raw/metadata.csv` file contains the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `image_id` | int | Unique image identifier | 0, 1, 2, ... |
| `identity` | str | Individual animal ID (NaN for test set) | `LynxID2025_lynx_37` |
| `path` | str | **Relative path from data/raw/** | `images/LynxID2025/train/37/000f9ee1...jpg` |
| `date` | str | Capture date (may be NaN) | `2024-01-15` |
| `orientation` | str | Animal orientation (left, right, back, etc.) | `right` |
| `species` | str | Animal species | `lynx` |
| `split` | str | Train/test split | `train` or `test` |
| `dataset` | str | Source dataset name | `LynxID2025` |

## Important Notes

### Path Column
⚠️ **The column is named `path`, NOT `image_path`**

The `path` column already includes the `images/` directory, so to load an image:
```python
from pathlib import Path
img_path = Path('data/raw') / row['path']
```

**Correct:**
```python
row['path']  # ✓ Returns: 'images/LynxID2025/train/37/...'
```

**Incorrect:**
```python
row['image_path']  # ✗ KeyError: 'image_path'
```

### Dataset Statistics

```
Total images: 15,483
- Train images: 13,074 (with identity labels)
- Test images: 2,409 (identity is NaN)

Datasets:
- SeaTurtleID2022: 9,229 images
- LynxID2025: 3,903 images
- SalamanderID2025: 2,077 images
- TexasHornedLizards: 274 images

Unique identities: 1,102
```

## Sample Submission Format

The `data/raw/sample_submission.csv` file has:

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | int | Test image ID |
| `cluster` | str | Predicted cluster/identity |

Shape: (2,409, 2) - one row per test image

## Loading Data

### Using AnimalCLEFDataset

```python
from src.data_loading import AnimalCLEFDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = AnimalCLEFDataset(
    metadata_path='data/raw/metadata.csv',
    data_dir='data/raw',  # Base directory
    transform=transform
)

image, info = dataset[0]
```

### Manual Loading

```python
import pandas as pd
from pathlib import Path
from PIL import Image

metadata = pd.read_csv('data/raw/metadata.csv')
row = metadata.iloc[0]

# Load image
img_path = Path('data/raw') / row['path']
image = Image.open(img_path)
```

## Common Mistakes to Avoid

1. ❌ Using `row['image_path']` instead of `row['path']`
2. ❌ Not including 'data/raw' as the base path
3. ❌ Trying to access identity for test images (it's NaN)
4. ❌ Assuming all metadata fields are always present (some have NaN)

## Quick Checks

```python
import pandas as pd

# Load and inspect
metadata = pd.read_csv('data/raw/metadata.csv')

print(f"Columns: {metadata.columns.tolist()}")
print(f"Total images: {len(metadata)}")
print(f"Train images: {len(metadata[metadata['split'] == 'train'])}")
print(f"Test images: {len(metadata[metadata['split'] == 'test'])}")
print(f"Missing identities: {metadata['identity'].isna().sum()}")

# Check path format
print(f"\nSample paths:")
print(metadata['path'].head(3).tolist())
```

Expected output:
```
Columns: ['image_id', 'identity', 'path', 'date', 'orientation', 'species', 'split', 'dataset']
Total images: 15483
Train images: 13074
Test images: 2409
Missing identities: 2409

Sample paths:
['images/LynxID2025/train/37/000f9ee1aad063a4485...',
 'images/LynxID2025/train/37/0020edb6689e9f78462...',
 'images/LynxID2025/train/49/003152e4145b5b69400...']
```
