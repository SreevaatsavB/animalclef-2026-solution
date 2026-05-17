#!/usr/bin/env python3
"""
build_notebook_v1_2.py
======================
Reads V1 notebook, inserts SAM 3 cells after Cell 1.5, patches local
feature extractors (SIFT / ALIKED / DISK) with resolve_image_path(),
and writes the V1.2 notebook.

Usage:
    python3 build_notebook_v1_2.py
"""

import json
import copy
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_NB  = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v1/ensemble_global_local_reid.ipynb')
OUT_DIR = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v1_2')
OUT_NB  = OUT_DIR / 'ensemble_global_local_reid_v1_2.ipynb'

# ── New cell sources ───────────────────────────────────────────────────────────

CELL_1_6_SOURCE = """\
# Cell 1.6: SAM 3 — build SEG_MAP from pre-segmented cache
#
# NOTE: Cell 1.1 pins transformers==4.36.0 for MiewID v3 compatibility.
# Loading Sam3Model requires transformers>=4.48.0, which conflicts.
# We avoid that conflict entirely by using ONLY the pre-segmented images
# already stored in the animalclef-26-sam3 Kaggle dataset.
# Any image not in the cache falls back to the original (no live SAM 3).
from pathlib import Path

SEG_CACHE_ROOT = Path('/kaggle/input/datasets/sreevaatsavbavana/animalclef-26-sam3/sam3_yolo_output/segmented_images')
_root = Path(ROOT_DIR)  # ROOT_DIR is a str defined in Cell 1.4

# Key by relative path (e.g. 'LynxID2025/img001.jpg') not stem, to avoid
# collisions when two datasets contain files with identical filenames.
SEG_MAP = {}  # {str(row['path']): Path_to_segmented_jpg}

all_meta = pd.concat([train_meta, test_meta])
for _, row in tqdm(all_meta.iterrows(), total=len(all_meta), desc='Building SEG_MAP'):
    stem    = Path(row['path']).stem   # e.g. 'img001'
    ds_name = row['dataset']
    rel_key = str(row['path'])         # e.g. 'LynxID2025/img001.jpg'

    cached = SEG_CACHE_ROOT / ds_name / (stem + '.jpg')
    if cached.exists():
        SEG_MAP[rel_key] = cached

n_total    = len(all_meta)
n_cached   = len(SEG_MAP)
n_fallback = n_total - n_cached
print(f'SEG_MAP built: {n_cached:,} cached  |  {n_fallback:,} fallback to original  |  {n_total:,} total')

# Per-species and per-split breakdown
print()
print(f'{"Dataset":<25} {"split":<6} {"cached":>7} {"total":>7} {"coverage":>9}')
print('-' * 58)
for ds in sorted(all_meta["dataset"].unique()):
    for split, meta_split in [("train", train_meta), ("test", test_meta)]:
        rows = meta_split[meta_split["dataset"] == ds]
        if len(rows) == 0:
            continue
        n_hit = sum(1 for _, r in rows.iterrows() if str(r["path"]) in SEG_MAP)
        pct = 100 * n_hit / len(rows)
        flag = "" if pct == 100 else " ⚠" if pct < 50 else " ✓"
        print(f'  {ds:<23} {split:<6} {n_hit:>7,} {len(rows):>7,} {pct:>8.1f}%{flag}')

def resolve_image_path(img_path):
    \"\"\"Return segmented image path if available, else original.
    Looks up by relative path from ROOT_DIR to avoid stem collisions.\"\"\"
    p = Path(img_path)
    try:
        rel_key = str(p.relative_to(_root))
    except ValueError:
        rel_key = None
    if rel_key and rel_key in SEG_MAP:
        return SEG_MAP[rel_key]
    return p

print('resolve_image_path() ready.')
"""

# ── Patch strings for extractor extract() methods ─────────────────────────────
# These are the exact first two lines of each extract() method in V1.
# We replace with the same lines + the resolve_image_path() call inserted first.

PATCHES = [
    # (marker_string, replacement_string)
    # Cell 3.1 — SIFTExtractor.extract
    (
        '    def extract(self, image_path):\n        """Extract SIFT keypoints and descriptors."""\n        img = cv2.imread(str(image_path))',
        '    def extract(self, image_path):\n        """Extract SIFT keypoints and descriptors."""\n        image_path = resolve_image_path(image_path)  # SAM 3 segmentation\n        img = cv2.imread(str(image_path))',
    ),
    # Cell 3.3 — ALIKEDExtractor.extract
    (
        '    def extract(self, image_path):\n        """Extract ALIKED keypoints and descriptors."""\n        img = cv2.imread(str(image_path))',
        '    def extract(self, image_path):\n        """Extract ALIKED keypoints and descriptors."""\n        image_path = resolve_image_path(image_path)  # SAM 3 segmentation\n        img = cv2.imread(str(image_path))',
    ),
    # Cell 3.4 — DISKExtractor.extract
    (
        '    def extract(self, image_path):\n        """Extract DISK keypoints and descriptors."""\n        img = cv2.imread(str(image_path))',
        '    def extract(self, image_path):\n        """Extract DISK keypoints and descriptors."""\n        image_path = resolve_image_path(image_path)  # SAM 3 segmentation\n        img = cv2.imread(str(image_path))',
    ),
]


def make_code_cell(source: str) -> dict:
    """Create a minimal Jupyter code cell dict."""
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source,
    }


def find_cell_index(cells: list, marker: str) -> int:
    """Return index of first cell whose source contains marker."""
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        if marker in src:
            return i
    return -1


def patch_cell_source(cells: list, old_str: str, new_str: str) -> bool:
    """
    Find the cell containing old_str and replace it with new_str.
    Returns True if the patch was applied.
    """
    for cell in cells:
        src = ''.join(cell['source'])
        if old_str in src:
            patched = src.replace(old_str, new_str, 1)
            # Store back as single string (Jupyter accepts both str and list)
            cell['source'] = patched
            return True
    return False


def main():
    print(f'Reading source notebook: {SRC_NB}')
    with open(SRC_NB) as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f'  Source cells: {len(cells)}')

    # ── 1. Find Cell 1.5 (dataset loading) ────────────────────────────────────
    idx_1_5 = find_cell_index(cells, 'Cell 1.5: Load Data')
    if idx_1_5 == -1:
        raise RuntimeError('Could not find "Cell 1.5: Load Data" in source notebook.')
    print(f'  Found Cell 1.5 at index {idx_1_5}')

    # ── 2. Insert Cell 1.6 after 1.5 ─────────────────────────────────────────
    # (No separate pip-install cell needed — SAM 3 model not loaded at runtime.
    #  Cell 1.6 only builds SEG_MAP from the pre-segmented cache on disk.)
    cell_1_6 = make_code_cell(CELL_1_6_SOURCE)
    cells.insert(idx_1_5 + 1, cell_1_6)
    print(f'  Inserted Cell 1.6 after index {idx_1_5}')

    # ── 3. Patch local feature extractor extract() methods ────────────────────
    for i, (old, new) in enumerate(PATCHES, 1):
        applied = patch_cell_source(cells, old, new)
        if applied:
            print(f'  Patch {i}/3 applied OK')
        else:
            print(f'  WARNING: Patch {i}/3 NOT applied — marker not found!')

    # ── 4. Update kernel/language metadata ────────────────────────────────────
    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }

    print(f'  Total cells after edits: {len(cells)}')

    # ── 5. Write output notebook ───────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_NB, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'\nOutput notebook written: {OUT_NB}')

    # ── 6. Write kernel-metadata.json (Kaggle metadata) ───────────────────────
    kernel_meta = {
        'id': 'sreevaatsavbavana/animalclef-26-v1-2',
        'title': 'AnimalCLEF 26 V1.2 (SAM3 segmentation)',
        'code_file': 'ensemble_global_local_reid_v1_2.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': [
            'sreevaatsavbavana/animalclef-26-sam3',
        ],
        'competition_sources': [
            'animal-clef-2026',
        ],
        'kernel_sources': [],
        'model_sources': [],
    }
    km_path = OUT_DIR / 'kernel-metadata.json'
    with open(km_path, 'w') as f:
        json.dump(kernel_meta, f, indent=2)
    print(f'Kaggle kernel-metadata.json written: {km_path}')

    # ── 7. Verification summary ────────────────────────────────────────────────
    print('\n── Verification ──────────────────────────────────────────────────────')
    with open(OUT_NB) as f:
        nb_check = json.load(f)
    all_src = '\n'.join(''.join(c['source']) for c in nb_check['cells'])

    checks = [
        ('Cell 1.6 present',            'Cell 1.6: SAM 3'                      in all_src),
        ('SEG_MAP built print',         'SEG_MAP built'                         in all_src),
        ('No SAM3 model load',           'Sam3Model.from_pretrained' not in all_src and
                                         'AutoProcessor.from_pretrained' not in all_src),
        ('No tqdm re-import',           'from tqdm import tqdm' not in all_src),
        ('ROOT_DIR str-safe',           'Path(ROOT_DIR)' in all_src),
        ('SEG_MAP rel_key',             'rel_key' in all_src),
        ('resolve_image_path def',      'def resolve_image_path'                in all_src),
        ('SIFT patch applied',          'SIFTExtractor' in all_src and
                                        'resolve_image_path(image_path)' in all_src),
        ('ALIKED patch applied',        'ALIKEDExtractor' in all_src),
        ('DISK patch applied',          'DISKExtractor' in all_src),
        ('SEG_CACHE_ROOT set',          'SEG_CACHE_ROOT' in all_src),
    ]

    all_ok = True
    for name, result in checks:
        status = '✓' if result else '✗'
        if not result:
            all_ok = False
        print(f'  {status} {name}')

    if all_ok:
        print('\nAll checks passed. V1.2 notebook is ready.')
    else:
        print('\nSome checks FAILED — review output above.')


if __name__ == '__main__':
    main()
