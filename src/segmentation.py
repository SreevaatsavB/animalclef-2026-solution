"""
SAM-2 segmentation utilities for wildlife images.
Removes background clutter to improve embedding quality.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path


def load_sam2_model(checkpoint_path="checkpoints/sam2.1_hiera_large.pt", device=None):
    """
    Load SAM-2 model for segmentation.

    Args:
        checkpoint_path: Path to SAM-2 checkpoint
        device: Device (mps/cuda/cpu), auto-detected if None

    Returns:
        SAM-2 predictor ready for inference
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"Loading SAM-2 on {device}...")

    # Import SAM-2 components
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Download checkpoint if not exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SAM-2 checkpoint...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"✓ Downloaded to {checkpoint_path}")

    # Load model
    sam2_checkpoint = str(checkpoint_path)
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Try to load from SAM2 package
    try:
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)
    except FileNotFoundError:
        # Try with absolute config path from package
        import sam2
        package_root = Path(sam2.__file__).parent
        model_cfg = package_root / "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2 = build_sam2(str(model_cfg), sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2)

    return predictor


def segment_animal(image: Image.Image, predictor, return_largest_mask=True):
    """
    Segment animal in image using SAM-2 automatic mask generation.

    Args:
        image: PIL Image (RGB)
        predictor: SAM-2 predictor
        return_largest_mask: If True, return only the largest mask by area

    Returns:
        mask: Binary mask (H, W) as numpy array
        bbox: Bounding box [x1, y1, x2, y2]
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Convert PIL to numpy
    image_np = np.array(image)

    # Use automatic mask generation to find ALL objects
    # This avoids the issue of single-point prompts missing parts of the animal
    mask_generator = SAM2AutomaticMaskGenerator(
        predictor.model,
        points_per_side=16,  # 16x16=256 points (balanced precision)
        pred_iou_thresh=0.80,  # Balanced threshold for quality
        stability_score_thresh=0.88,  # Balanced threshold for stability
        crop_n_layers=0,  # Disable cropping for speed
        min_mask_region_area=150,  # Filter out tiny regions
    )

    # Generate all masks
    masks_data = mask_generator.generate(image_np)

    if len(masks_data) == 0:
        # Fallback: no masks found, return full image
        h, w = image_np.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        bbox = [0, 0, w, h]
        return mask, bbox

    # Select the largest mask by area (most likely the main animal)
    if return_largest_mask:
        largest_mask_data = max(masks_data, key=lambda x: x['area'])
        mask = largest_mask_data['segmentation']
        bbox = largest_mask_data['bbox']  # [x, y, w, h] format
        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
    else:
        # Return the first mask
        mask = masks_data[0]['segmentation']
        bbox_xywh = masks_data[0]['bbox']
        bbox = [int(bbox_xywh[0]), int(bbox_xywh[1]),
                int(bbox_xywh[0] + bbox_xywh[2]), int(bbox_xywh[1] + bbox_xywh[3])]

    # Detect and fix inverted masks
    # Check if mask touches all 4 edges (strong indicator of background)
    h, w = mask.shape
    touches_top = mask[0, :].any()
    touches_bottom = mask[-1, :].any()
    touches_left = mask[:, 0].any()
    touches_right = mask[:, -1].any()
    touches_all_edges = touches_top and touches_bottom and touches_left and touches_right

    # Check if bbox covers most of the image (close-up shot)
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = h * w
    bbox_coverage = bbox_area / image_area

    # Also check mask ratio
    mask_ratio = mask.sum() / (h * w)
    touches_count = sum([touches_top, touches_bottom, touches_left, touches_right])

    # Inversion logic:
    # - If mask is 55-75% AND touches all 4 edges → likely background, flip it
    # - If mask is >75% → likely close-up animal filling frame, don't flip
    # - If mask touches all edges but is <55% → already inverted, don't flip again

    should_flip = False
    if touches_all_edges and 0.55 <= mask_ratio <= 0.75:
        # Partial background touching all edges
        should_flip = True
    elif mask_ratio > 0.7 and touches_count >= 3 and mask_ratio < 0.75:
        # Large mask touching most edges (but not a close-up)
        should_flip = True

    if should_flip:
        # Mask is inverted - flip it
        mask = ~mask
        # Recalculate bbox for inverted mask
        bbox = mask_to_bbox(mask)

    return mask, bbox


def mask_to_bbox(mask):
    """Convert binary mask to bounding box."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        # Empty mask - return full image
        return [0, 0, mask.shape[1], mask.shape[0]]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)]


def visualize_segmentation(image, mask, bbox=None, save_path=None):
    """
    Visualize segmentation result.

    Args:
        image: Original PIL Image
        mask: Binary mask
        bbox: Optional bounding box [x1, y1, x2, y2]
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Mask overlay
    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=0.5, cmap='jet')
    if bbox:
        x1, y1, x2, y2 = bbox
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1,
                         fill=False, edgecolor='red', linewidth=2)
        axes[1].add_patch(rect)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')

    # Cropped result
    if bbox:
        x1, y1, x2, y2 = bbox
        cropped = image.crop((x1, y1, x2, y2))
        axes[2].imshow(cropped)
        axes[2].set_title(f"Cropped ({x2-x1}x{y2-y1})")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
