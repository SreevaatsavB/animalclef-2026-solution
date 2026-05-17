#!/usr/bin/env python3
"""
build_notebook_v1_3.py
======================
V1.3 approach: keypoint-mask filtering (NOT white-background replacement).

V1.2 painted background pixels white → SIFT found strong edges at the
animal/white boundary → those boundary keypoints don't match across
different images → score dropped 0.30655 → 0.26330.

V1.3 fix:
  - SIFT runs on the ORIGINAL image (no painting, no artifacts)
  - After detection, keypoints whose (x, y) falls on background pixels
    are discarded using the binary mask derived from the segmented image
  - Only animal-region keypoints are used for matching

LynxID2025 has 0% cache coverage → no change vs V1 for that species.
Other three species get cleaner SIFT features with fewer background KPs.

Usage:
    python3 build_notebook_v1_3.py
"""

import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_NB  = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v1/ensemble_global_local_reid.ipynb')
OUT_DIR = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v1_3')
OUT_NB  = OUT_DIR / 'ensemble_global_local_reid_v1_3.ipynb'

# ── Cell 1.6: SEG_MAP + get_seg_mask() ────────────────────────────────────────
CELL_1_6_SOURCE = """\
# Cell 1.6: SAM 3 — build SEG_MAP and keypoint-mask helper
#
# Strategy (V1.3): run SIFT on the ORIGINAL image, then discard keypoints
# that land on background pixels.  This avoids the white-boundary artifacts
# that hurt V1.2 (score 0.26330 vs V1 baseline 0.30655).
#
# LynxID2025 has 0% cache coverage → get_seg_mask() returns None → no change
# for that species.  All other species get filtered keypoints.
from pathlib import Path
import cv2
import numpy as np

SEG_CACHE_ROOT = Path('/kaggle/input/datasets/sreevaatsavbavana/animalclef-26-sam3/sam3_yolo_output/segmented_images')
_root = Path(ROOT_DIR)  # ROOT_DIR defined in Cell 1.4 as a str

# Key: relative path  e.g. 'SeaTurtleID2022/img001.jpg'
SEG_MAP = {}  # {rel_key: Path_to_segmented_jpg}

all_meta = pd.concat([train_meta, test_meta])
for _, row in tqdm(all_meta.iterrows(), total=len(all_meta), desc='Building SEG_MAP'):
    stem    = Path(row['path']).stem
    ds_name = row['dataset']
    rel_key = str(row['path'])
    cached  = SEG_CACHE_ROOT / ds_name / (stem + '.jpg')
    if cached.exists():
        SEG_MAP[rel_key] = cached

n_total    = len(all_meta)
n_cached   = len(SEG_MAP)
n_fallback = n_total - n_cached
print(f'SEG_MAP built: {n_cached:,} cached  |  {n_fallback:,} fallback to original  |  {n_total:,} total')

# Per-species / per-split breakdown
print()
print(f'{"Dataset":<25} {"split":<6} {"cached":>7} {"total":>7} {"coverage":>9}')
print('-' * 58)
for ds in sorted(all_meta["dataset"].unique()):
    for split, meta_split in [("train", train_meta), ("test", test_meta)]:
        rows = meta_split[meta_split["dataset"] == ds]
        if len(rows) == 0:
            continue
        n_hit = sum(1 for _, r in rows.iterrows() if str(r["path"]) in SEG_MAP)
        pct   = 100 * n_hit / len(rows)
        flag  = "" if pct == 100 else " ⚠" if pct < 50 else " ✓"
        print(f'  {ds:<23} {split:<6} {n_hit:>7,} {len(rows):>7,} {pct:>8.1f}%{flag}')

def get_seg_mask(img_path):
    \"\"\"
    Return a binary uint8 mask (1=animal, 0=background) derived from the
    pre-segmented image, or None if not in the cache.

    The segmented image has background pixels set to pure white (255,255,255).
    Mask = pixels that are NOT pure white.  Imperfect for white-furred/scaled
    animals, but good enough for SeaTurtles, Salamanders, and TexasLizards.
    \"\"\"
    p = Path(img_path)
    try:
        rel_key = str(p.relative_to(_root))
    except ValueError:
        return None
    if rel_key not in SEG_MAP:
        return None
    seg = cv2.imread(str(SEG_MAP[rel_key]))
    if seg is None:
        return None
    # Background was painted to (255, 255, 255) — mark those as 0
    is_bg = (seg[:, :, 0] == 255) & (seg[:, :, 1] == 255) & (seg[:, :, 2] == 255)
    return (~is_bg).astype(np.uint8)

print('get_seg_mask() ready.')
"""

# ── SIFT patch: keypoint filtering by mask ────────────────────────────────────
# Marker: the block between detectAndCompute and the early-return check.
# We insert the mask-filter block right after the existing early-return.

_SIFT_OLD = (
    '        if descriptors is None or len(keypoints) < 4:\n'
    '            return None\n'
    '        \n'
    '        # Convert keypoints to numpy array (x, y)\n'
    '        kpts_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)'
)

_SIFT_NEW = (
    '        if descriptors is None or len(keypoints) < 4:\n'
    '            return None\n'
    '        \n'
    '        # Filter keypoints to animal region using segmentation mask.\n'
    '        # SIFT runs on the ORIGINAL image (no white-bg boundary artifacts).\n'
    '        # get_seg_mask() returns None for uncached images → all kpts kept.\n'
    '        seg_mask = get_seg_mask(image_path)\n'
    '        if seg_mask is not None:\n'
    '            h, w = seg_mask.shape\n'
    '            keep = [\n'
    '                i for i, kp in enumerate(keypoints)\n'
    '                if 0 <= int(kp.pt[1]) < h\n'
    '                and 0 <= int(kp.pt[0]) < w\n'
    '                and seg_mask[int(kp.pt[1]), int(kp.pt[0])] == 1\n'
    '            ]\n'
    '            if len(keep) >= 4:\n'
    '                keypoints   = [keypoints[i] for i in keep]\n'
    '                descriptors = descriptors[keep]\n'
    '        \n'
    '        # Convert keypoints to numpy array (x, y)\n'
    '        kpts_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)'
)

PATCHES = [
    (_SIFT_OLD, _SIFT_NEW),  # Cell 3.1 — SIFTExtractor only (ALIKED/DISK unused)
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_code_cell(source: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source,
    }


def find_cell_index(cells: list, marker: str) -> int:
    for i, cell in enumerate(cells):
        if marker in ''.join(cell['source']):
            return i
    return -1


def patch_cell_source(cells: list, old_str: str, new_str: str) -> bool:
    for cell in cells:
        src = ''.join(cell['source'])
        if old_str in src:
            cell['source'] = src.replace(old_str, new_str, 1)
            return True
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'Reading source notebook: {SRC_NB}')
    with open(SRC_NB) as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f'  Source cells: {len(cells)}')

    # 1. Insert Cell 1.6 after Cell 1.5
    idx_1_5 = find_cell_index(cells, 'Cell 1.5: Load Data')
    if idx_1_5 == -1:
        raise RuntimeError('Cell 1.5 not found')
    cells.insert(idx_1_5 + 1, make_code_cell(CELL_1_6_SOURCE))
    print(f'  Inserted Cell 1.6 after index {idx_1_5}')

    # 2. Patch SIFT extractor
    for i, (old, new) in enumerate(PATCHES, 1):
        ok = patch_cell_source(cells, old, new)
        print(f'  Patch {i}/{len(PATCHES)} {"OK" if ok else "FAILED — marker not found!"}')

    # 3. Kernel metadata
    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    print(f'  Total cells: {len(cells)}')

    # 4. Write notebook
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_NB, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'\nOutput notebook: {OUT_NB}')

    # 5. Kaggle kernel-metadata.json
    km = {
        'id': 'sreevaatsavbavana/animalclef-26-v1-3',
        'title': 'AnimalCLEF 26 V1.3 (SAM3 keypoint mask)',
        'code_file': 'ensemble_global_local_reid_v1_3.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': ['sreevaatsavbavana/animalclef-26-sam3'],
        'competition_sources': ['animal-clef-2026'],
        'kernel_sources': [],
        'model_sources': [],
    }
    km_path = OUT_DIR / 'kernel-metadata.json'
    with open(km_path, 'w') as f:
        json.dump(km, f, indent=2)
    print(f'Kaggle kernel-metadata:  {km_path}')

    # 6. Verify
    print('\n── Verification ──────────────────────────────────────────────────────')
    with open(OUT_NB) as f:
        nb_check = json.load(f)
    all_src = '\n'.join(''.join(c['source']) for c in nb_check['cells'])

    checks = [
        ('Cell 1.6 present',        'Cell 1.6: SAM 3'              in all_src),
        ('SEG_MAP built',           'SEG_MAP built'                 in all_src),
        ('get_seg_mask defined',    'def get_seg_mask'              in all_src),
        ('No SAM3 model load',      'from_pretrained' not in all_src or
                                    'Sam3Model' not in all_src),
        ('No tqdm re-import',       'from tqdm import tqdm' not in all_src),
        ('ROOT_DIR str-safe',       'Path(ROOT_DIR)' in all_src),
        ('Keypoint filter patch',   'seg_mask = get_seg_mask' in all_src),
        ('Filter keeps >=4 kpts',   'len(keep) >= 4' in all_src),
        ('No white-bg replacement', 'resolve_image_path' not in all_src),
        ('SEG_CACHE_ROOT set',      'SEG_CACHE_ROOT' in all_src),
        ('Coverage table present',  'coverage' in all_src),
    ]

    all_ok = True
    for name, result in checks:
        status = '✓' if result else '✗'
        if not result:
            all_ok = False
        print(f'  {status} {name}')

    print('\nAll checks passed. V1.3 notebook is ready.' if all_ok
          else '\nSome checks FAILED.')


if __name__ == '__main__':
    main()
