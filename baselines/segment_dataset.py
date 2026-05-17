"""
Automated segmentation and cropping for AnimalCLEF datasets.
Removes background clutter to improve embedding quality.

Usage:
    python baselines/segment_dataset.py --species all
    python baselines/segment_dataset.py --species SalamanderID2025
    python baselines/segment_dataset.py --species SeaTurtleID2022 --margin 1.5
"""

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.segmentation import load_sam2_model, segment_animal


class SegmentationPipeline:
    def __init__(self, sam2_predictor, margin=1.2):
        """
        Args:
            sam2_predictor: Loaded SAM-2 predictor
            margin: Crop margin multiplier (1.2 = 20% padding)
        """
        self.predictor = sam2_predictor
        self.margin = margin
        self.species_configs = {
            'LynxID2025': {
                'description': 'Remove trees, foliage, snow',
                'min_animal_ratio': 0.15  # Animal should be ≥15% of image
            },
            'SalamanderID2025': {
                'description': 'Remove hands, ground, leaf litter',
                'min_animal_ratio': 0.10  # Salamanders can be small
            },
            'SeaTurtleID2022': {
                'description': 'Isolate head/carapace from water',
                'min_animal_ratio': 0.20
            },
            'TexasHornedLizards': {
                'description': 'Remove desert, dirt, vegetation',
                'min_animal_ratio': 0.12
            }
        }

    def segment_image(self, image_path: Path, species: str):
        """Segment single image and return cropped version."""
        try:
            # 1. Load image
            image = Image.open(image_path).convert('RGB')
            orig_w, orig_h = image.size

            # 2. Run SAM-2 segmentation
            mask, bbox = segment_animal(image, self.predictor)

            # 3. Validate mask quality
            animal_ratio = mask.sum() / (orig_w * orig_h)
            min_ratio = self.species_configs[species]['min_animal_ratio']

            if animal_ratio < min_ratio:
                # Mask too small - likely failed, return None
                return None, None, "mask_too_small"

            # 4. Extract bounding box with margin
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1

            # Add margin
            margin_w = int(w * (self.margin - 1) / 2)
            margin_h = int(h * (self.margin - 1) / 2)

            x1_crop = max(0, x1 - margin_w)
            y1_crop = max(0, y1 - margin_h)
            x2_crop = min(orig_w, x2 + margin_w)
            y2_crop = min(orig_h, y2 + margin_h)

            # 5. Crop image
            image_cropped = image.crop((x1_crop, y1_crop, x2_crop, y2_crop))

            # 6. Return cropped image + metadata
            crop_info = {
                'x1': x1_crop, 'y1': y1_crop,
                'x2': x2_crop, 'y2': y2_crop,
                'animal_ratio': float(animal_ratio),
                'orig_w': orig_w, 'orig_h': orig_h
            }

            return image_cropped, crop_info, "success"

        except Exception as e:
            return None, None, f"error: {str(e)}"

    def process_dataset(self, metadata_df, input_dir, output_dir, species):
        """Process all images for a given species."""
        output_dir = Path(output_dir)
        species_df = metadata_df[metadata_df['dataset'] == species].copy()

        results = []
        failed_count = 0

        print(f"\n{'='*70}")
        print(f"Processing {species}: {len(species_df)} images")
        print(f"Goal: {self.species_configs[species]['description']}")
        print(f"{'='*70}")

        for idx, row in tqdm(species_df.iterrows(), total=len(species_df), desc=f"{species}"):
            img_path = Path(input_dir) / row['path']

            if not img_path.exists():
                print(f"  ⚠️  File not found: {img_path}")
                failed_count += 1
                # Use placeholder
                crop_info = {'status': 'file_not_found'}
                results.append({
                    'image_id': row['image_id'],
                    'original_path': row['path'],
                    'cropped_path': row['path'],
                    **crop_info
                })
                continue

            # Segment and crop
            cropped_img, crop_info, status = self.segment_image(img_path, species)

            if status != "success":
                failed_count += 1
                # Fallback: copy original image
                cropped_img = Image.open(img_path).convert('RGB')
                crop_info = {'status': f'fallback_{status}'}

            # Save cropped image (maintain directory structure)
            rel_path = Path(row['path'])
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped_img.save(output_path, quality=95)

            # Record crop metadata
            results.append({
                'image_id': row['image_id'],
                'original_path': row['path'],
                'cropped_path': str(rel_path),
                **crop_info
            })

        print(f"\n✓ Processed: {len(species_df) - failed_count}/{len(species_df)}")
        print(f"⚠️  Fallback to original: {failed_count}")

        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Segment wildlife images using SAM-2 to remove background clutter'
    )
    parser.add_argument('--species', default='all',
                       choices=['all', 'LynxID2025', 'SalamanderID2025',
                               'SeaTurtleID2022', 'TexasHornedLizards'],
                       help='Species to process (default: all except Lynx)')
    parser.add_argument('--input-dir', default='data/raw',
                       help='Input directory containing raw images')
    parser.add_argument('--output-dir', default='data/processed/cropped',
                       help='Output directory for cropped images')
    parser.add_argument('--margin', type=float, default=1.2,
                       help='Crop margin multiplier (1.2 = 20%% padding)')
    parser.add_argument('--checkpoint', default='checkpoints/sam2.1_hiera_large.pt',
                       help='Path to SAM-2 checkpoint')
    args = parser.parse_args()

    print("="*70)
    print("AnimalCLEF 2026 - Automated Segmentation Pipeline")
    print("="*70)

    # Load SAM-2 model
    print("\nStep 1: Loading SAM-2 model...")
    sam2_predictor = load_sam2_model(checkpoint_path=args.checkpoint)
    print("✓ Model loaded successfully")

    # Load metadata
    print(f"\nStep 2: Loading metadata from {args.input_dir}/metadata.csv...")
    metadata = pd.read_csv(f'{args.input_dir}/metadata.csv')
    print(f"✓ Loaded {len(metadata)} image records")

    # Create segmentation pipeline
    pipeline = SegmentationPipeline(sam2_predictor, margin=args.margin)

    # Process datasets (exclude Lynx since it's already segmented)
    if args.species == 'all':
        species_list = ['SalamanderID2025', 'SeaTurtleID2022', 'TexasHornedLizards']
        print("\nNote: Skipping LynxID2025 (already segmented)")
    else:
        species_list = [args.species]

    all_results = []
    for species in species_list:
        results_df = pipeline.process_dataset(
            metadata, args.input_dir, args.output_dir, species
        )
        all_results.append(results_df)

    # Save crop metadata
    crop_metadata = pd.concat(all_results, ignore_index=True)
    metadata_path = f'{args.output_dir}/crop_metadata.csv'
    crop_metadata.to_csv(metadata_path, index=False)

    print("\n" + "="*70)
    print("✓ Segmentation Complete!")
    print("="*70)
    print(f"Cropped images saved to: {args.output_dir}/")
    print(f"Crop metadata saved to: {metadata_path}")
    print("\nNext steps:")
    print("  1. Validate segmentation quality:")
    print("     python baselines/validate_segmentation.py --species SalamanderID2025")
    print("  2. Re-run baseline with cropped images:")
    print("     python baselines/run_baseline.py (set USE_CROPPED=True)")
    print("="*70)


if __name__ == "__main__":
    main()
