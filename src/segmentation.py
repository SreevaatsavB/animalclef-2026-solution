import torch
import numpy as np
from PIL import Image
from pathlib import Path


def load_sam2_model(checkpoint_path="checkpoints/sam2.1_hiera_large.pt", device=None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SAM-2 on {device}...")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        print("Downloading SAM-2 checkpoint...")
        urllib.request.urlretrieve(url, checkpoint_path)

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    try:
        sam2 = build_sam2(model_cfg, str(checkpoint_path), device=device)
    except FileNotFoundError:
        import sam2 as sam2_pkg
        model_cfg = Path(sam2_pkg.__file__).parent / "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2 = build_sam2(str(model_cfg), str(checkpoint_path), device=device)

    return SAM2ImagePredictor(sam2)


def segment_animal(image, predictor, return_largest=True):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    img_np = np.array(image)
    mask_gen = SAM2AutomaticMaskGenerator(
        predictor.model,
        points_per_side=16,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        crop_n_layers=0,
        min_mask_region_area=150,
    )
    masks = mask_gen.generate(img_np)

    if not masks:
        h, w = img_np.shape[:2]
        return np.ones((h, w), dtype=bool), [0, 0, w, h]

    m = max(masks, key=lambda x: x['area']) if return_largest else masks[0]
    mask = m['segmentation']
    bx = m['bbox']
    bbox = [int(bx[0]), int(bx[1]), int(bx[0] + bx[2]), int(bx[1] + bx[3])]

    h, w = mask.shape
    mask_ratio = mask.sum() / (h * w)
    touches = [mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()]

    # if mask covers 55-75% AND touches all edges it's probably the background, not the animal
    if all(touches) and 0.55 <= mask_ratio <= 0.75:
        mask = ~mask
        bbox = mask_to_bbox(mask)

    return mask, bbox


def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, mask.shape[1], mask.shape[0]]
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [int(x0), int(y0), int(x1 + 1), int(y1 + 1)]


def visualize_segmentation(image, mask, bbox=None, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=0.5, cmap='jet')
    if bbox:
        x1, y1, x2, y2 = bbox
        axes[1].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))
    axes[1].set_title("Mask")
    axes[1].axis('off')

    if bbox:
        axes[2].imshow(image.crop(bbox))
        axes[2].set_title(f"Cropped ({bbox[2]-bbox[0]}x{bbox[3]-bbox[1]})")
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
