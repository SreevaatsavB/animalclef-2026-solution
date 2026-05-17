"""
Validate segmentation quality by visualizing samples.

Usage:
    python baselines/validate_segmentation.py --species SalamanderID2025 --num-samples 10
    python baselines/validate_segmentation.py --species SeaTurtleID2022 --num-samples 20
"""

import argparse
import random
from pathlib import Path
import pandas as pd
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.segmentation import load_sam2_model, segment_animal, visualize_segmentation


def main():
    parser = argparse.ArgumentParser(
        description='Validate SAM-2 segmentation quality on sample images'
    )
    parser.add_argument('--species', required=True,
                       choices=['SalamanderID2025', 'SeaTurtleID2022', 'TexasHornedLizards'],
                       help='Species to validate')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of random samples to visualize')
    parser.add_argument('--output-dir', default='data/processed/validation',
                       help='Output directory for validation visualizations')
    parser.add_argument('--checkpoint', default='checkpoints/sam2.1_hiera_large.pt',
                       help='Path to SAM-2 checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    print("="*70)
    print(f"Validating Segmentation Quality: {args.species}")
    print("="*70)

    # Set random seed
    random.seed(args.seed)

    # Load metadata
    print("\nLoading metadata...")
    metadata = pd.read_csv('data/raw/metadata.csv')
    species_df = metadata[metadata['dataset'] == args.species]
    print(f"✓ Found {len(species_df)} images for {args.species}")

    # Sample random images
    n_samples = min(args.num_samples, len(species_df))
    samples = species_df.sample(n=n_samples, random_state=args.seed)
    print(f"✓ Selected {n_samples} random samples")

    # Load SAM-2
    print("\nLoading SAM-2 model...")
    sam2_predictor = load_sam2_model(checkpoint_path=args.checkpoint)
    print("✓ Model loaded")

    # Process and visualize each sample
    output_dir = Path(args.output_dir) / args.species
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations...")
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        img_path = Path('data/raw') / row['path']

        if not img_path.exists():
            print(f"  ⚠️  [{idx}/{n_samples}] File not found: {img_path}")
            continue

        image = Image.open(img_path).convert('RGB')

        # Segment
        mask, bbox = segment_animal(image, sam2_predictor)

        # Calculate stats
        animal_ratio = mask.sum() / (image.size[0] * image.size[1])

        # Visualize
        save_path = output_dir / f"{row['image_id']}_segmentation.png"
        visualize_segmentation(image, mask, bbox, save_path)

        print(f"  ✓ [{idx}/{n_samples}] {row['image_id']}: "
              f"animal_ratio={animal_ratio:.2%}, bbox={bbox}")

    print(f"\n{'='*70}")
    print(f"✓ Validation Complete!")
    print(f"{'='*70}")
    print(f"Visualizations saved to: {output_dir}/")
    print(f"\nReview the images to check:")
    print(f"  - Does the mask cover the animal well?")
    print(f"  - Is the bounding box tight around the animal?")
    print(f"  - Is background successfully removed?")
    print(f"\nIf quality looks good, run:")
    print(f"  python baselines/segment_dataset.py --species {args.species}")
    print("="*70)


if __name__ == "__main__":
    main()
