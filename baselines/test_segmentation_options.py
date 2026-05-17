"""
Test all segmentation output options on sample images.

Generates 4 types of outputs:
1. Bounding box crop (current approach)
2. Original + binary mask saved separately
3. Masked image (background removed)
4. All of the above

Usage:
    python baselines/test_segmentation_options.py
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from src.segmentation import load_sam2_model, segment_animal


def save_all_outputs(image, mask, bbox, output_dir, image_id, margin=1.2):
    """
    Generate all 4 segmentation output types.

    Args:
        image: PIL Image
        mask: Binary mask (H, W) numpy array
        bbox: Bounding box [x1, y1, x2, y2]
        output_dir: Directory to save outputs
        image_id: Image identifier
        margin: Crop margin multiplier
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_w, orig_h = image.size
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Calculate crop region with margin
    margin_w = int(w * (margin - 1) / 2)
    margin_h = int(h * (margin - 1) / 2)
    x1_crop = max(0, x1 - margin_w)
    y1_crop = max(0, y1 - margin_h)
    x2_crop = min(orig_w, x2 + margin_w)
    y2_crop = min(orig_h, y2 + margin_h)

    # === OPTION 1: Bounding box crop (current) ===
    crop_bbox = image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
    crop_bbox.save(output_dir / f"{image_id}_option1_bbox_crop.jpg", quality=95)

    # === OPTION 2: Original + Binary mask ===
    # Save original
    image.save(output_dir / f"{image_id}_option2_original.jpg", quality=95)
    # Save binary mask as PNG
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(output_dir / f"{image_id}_option2_mask.png")

    # === OPTION 3: Masked image (background removed) ===
    image_np = np.array(image)
    # Create 3-channel mask
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    # Apply mask (background becomes black)
    masked_image_np = image_np * mask_3ch
    masked_image = Image.fromarray(masked_image_np.astype(np.uint8))
    # Crop to bbox
    masked_crop = masked_image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
    masked_crop.save(output_dir / f"{image_id}_option3_masked.jpg", quality=95)

    # === OPTION 4: Combined visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title("Binary Mask")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image)
    axes[0, 2].imshow(mask, alpha=0.5, cmap='jet')
    from matplotlib.patches import Rectangle
    rect = Rectangle((x1_crop, y1_crop), x2_crop-x1_crop, y2_crop-y1_crop,
                     fill=False, edgecolor='red', linewidth=2)
    axes[0, 2].add_patch(rect)
    axes[0, 2].set_title("Mask Overlay + Bbox")
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(crop_bbox)
    axes[1, 0].set_title(f"Option 1: Bbox Crop\n({x2_crop-x1_crop}x{y2_crop-y1_crop})")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(masked_image)
    axes[1, 1].set_title("Option 3: Masked Full Image\n(background removed)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(masked_crop)
    axes[1, 2].set_title(f"Option 3: Masked Crop\n({x2_crop-x1_crop}x{y2_crop-y1_crop})")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"{image_id}_option4_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Return file sizes for comparison
    sizes = {
        'option1_bbox_crop': (output_dir / f"{image_id}_option1_bbox_crop.jpg").stat().st_size,
        'option2_original': (output_dir / f"{image_id}_option2_original.jpg").stat().st_size,
        'option2_mask': (output_dir / f"{image_id}_option2_mask.png").stat().st_size,
        'option3_masked': (output_dir / f"{image_id}_option3_masked.jpg").stat().st_size,
        'option4_comparison': (output_dir / f"{image_id}_option4_comparison.png").stat().st_size,
    }

    return sizes


def main():
    print("="*70)
    print("Testing Segmentation Output Options")
    print("="*70)

    # Load metadata
    print("\nLoading metadata...")
    metadata = pd.read_csv('data/raw/metadata.csv')

    # Sample more images from each species
    species_list = ['SalamanderID2025', 'TexasHornedLizards']
    samples_per_species = 5

    all_samples = []
    for species in species_list:
        species_df = metadata[metadata['dataset'] == species]
        samples = species_df.sample(n=samples_per_species, random_state=42)
        all_samples.append(samples)

    samples_df = pd.concat(all_samples, ignore_index=True)
    print(f"✓ Selected {len(samples_df)} sample images")

    # Load SAM-2
    print("\nLoading SAM-2 model...")
    sam2_predictor = load_sam2_model()
    print("✓ Model loaded")

    # Process each sample
    output_base = Path('data/processed/segmentation_tests')
    all_sizes = []

    print("\n" + "="*70)
    print("Generating all output types for each sample...")
    print("="*70)

    for idx, row in samples_df.iterrows():
        species = row['dataset']
        image_id = row['image_id']
        img_path = Path('data/raw') / row['path']

        print(f"\n[{idx+1}/{len(samples_df)}] Processing {species} - {image_id}")

        if not img_path.exists():
            print(f"  ⚠️  File not found: {img_path}")
            continue

        # Load and segment
        image = Image.open(img_path).convert('RGB')
        mask, bbox = segment_animal(image, sam2_predictor)

        # Calculate stats
        animal_ratio = mask.sum() / (image.size[0] * image.size[1])

        # Debug: check edge touching
        h, w = mask.shape
        touches_top = mask[0, :].any()
        touches_bottom = mask[-1, :].any()
        touches_left = mask[:, 0].any()
        touches_right = mask[:, -1].any()
        edges = f"T{int(touches_top)}B{int(touches_bottom)}L{int(touches_left)}R{int(touches_right)}"

        print(f"  Animal ratio: {animal_ratio:.2%}, Edges: {edges}")
        print(f"  Bbox: {bbox}")

        # Generate all outputs
        output_dir = output_base / species
        sizes = save_all_outputs(image, mask, bbox, output_dir, image_id)

        # Record sizes
        sizes['species'] = species
        sizes['image_id'] = image_id
        sizes['animal_ratio'] = animal_ratio
        all_sizes.append(sizes)

        print(f"  ✓ Saved all outputs to: {output_dir}/")

    # Summary
    print("\n" + "="*70)
    print("Summary: File Sizes")
    print("="*70)

    sizes_df = pd.DataFrame(all_sizes)

    for species in species_list:
        species_sizes = sizes_df[sizes_df['species'] == species]
        if len(species_sizes) == 0:
            continue

        print(f"\n{species}:")
        print(f"  Option 1 (Bbox Crop):      {species_sizes['option1_bbox_crop'].mean()/1024:.1f} KB")
        print(f"  Option 2 (Original):       {species_sizes['option2_original'].mean()/1024:.1f} KB")
        print(f"  Option 2 (Mask PNG):       {species_sizes['option2_mask'].mean()/1024:.1f} KB")
        print(f"  Option 3 (Masked Crop):    {species_sizes['option3_masked'].mean()/1024:.1f} KB")
        print(f"  Option 4 (Comparison):     {species_sizes['option4_comparison'].mean()/1024:.1f} KB")
        print(f"  Total per image:           {(species_sizes['option1_bbox_crop'] + species_sizes['option2_original'] + species_sizes['option2_mask'] + species_sizes['option3_masked']).mean()/1024:.1f} KB")

    print("\n" + "="*70)
    print("✓ Test Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_base}/")
    print("\nReview the outputs:")
    for species in species_list:
        print(f"  {output_base}/{species}/")
    print("\nFiles generated per image:")
    print("  *_option1_bbox_crop.jpg      - Bounding box crop")
    print("  *_option2_original.jpg       - Original image")
    print("  *_option2_mask.png           - Binary mask")
    print("  *_option3_masked.jpg         - Background removed")
    print("  *_option4_comparison.png     - Visual comparison of all options")
    print("="*70)


if __name__ == "__main__":
    main()
