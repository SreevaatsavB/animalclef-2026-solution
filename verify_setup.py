#!/usr/bin/env python3
"""
Verification script for AnimalCLEF 2026 setup
Run this to ensure everything is installed correctly
"""

import sys
from pathlib import Path

def check_packages():
    """Check if all required packages are installed"""
    print("=" * 60)
    print("CHECKING INSTALLED PACKAGES")
    print("=" * 60)

    packages = {
        'wildlife_datasets': 'WildlifeDatasets',
        'wildlife_tools': 'WildlifeTools',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm',
        'jupyter': 'Jupyter',
    }

    all_installed = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} - Installed")
        except ImportError:
            print(f"✗ {name:20s} - NOT FOUND")
            all_installed = False

    return all_installed


def check_data():
    """Check if competition data is available"""
    print("\n" + "=" * 60)
    print("CHECKING COMPETITION DATA")
    print("=" * 60)

    data_dir = Path('data/raw')
    required_files = [
        'metadata.csv',
        'sample_submission.csv',
        'images'
    ]

    all_present = True
    for file in required_files:
        filepath = data_dir / file
        if filepath.exists():
            if filepath.is_dir():
                num_items = len(list(filepath.iterdir()))
                print(f"✓ {file:30s} - Found ({num_items} items)")
            else:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✓ {file:30s} - Found ({size_mb:.2f} MB)")
        else:
            print(f"✗ {file:30s} - NOT FOUND")
            all_present = False

    return all_present


def check_directories():
    """Check if all project directories exist"""
    print("\n" + "=" * 60)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 60)

    directories = [
        'data/raw',
        'data/processed',
        'data/wildlife_datasets',
        'notebooks',
        'src',
        'experiments',
        'submissions'
    ]

    all_exist = True
    for directory in directories:
        dirpath = Path(directory)
        if dirpath.exists():
            print(f"✓ {directory:30s} - Exists")
        else:
            print(f"✗ {directory:30s} - NOT FOUND")
            all_exist = False

    return all_exist


def check_custom_modules():
    """Check if custom modules can be imported"""
    print("\n" + "=" * 60)
    print("CHECKING CUSTOM MODULES")
    print("=" * 60)

    sys.path.append('src')

    modules = [
        'data_loading',
        'models',
        'training',
        'inference'
    ]

    all_working = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ src/{module}.py - OK")
        except Exception as e:
            print(f"✗ src/{module}.py - ERROR: {e}")
            all_working = False

    return all_working


def check_pytorch_device():
    """Check PyTorch device availability"""
    print("\n" + "=" * 60)
    print("CHECKING PYTORCH DEVICE")
    print("=" * 60)

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

    print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")

    # Determine default device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\nRecommended device: {device}")
    return True


def main():
    """Run all verification checks"""
    print("\n" + "=" * 60)
    print("AnimalCLEF 2026 - Setup Verification")
    print("=" * 60 + "\n")

    results = {
        'Packages': check_packages(),
        'Data': check_data(),
        'Directories': check_directories(),
        'Custom Modules': check_custom_modules(),
        'PyTorch Device': check_pytorch_device()
    }

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check:20s}: {status}")

    if all(results.values()):
        print("\n🎉 All checks passed! You're ready to start competing!")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
