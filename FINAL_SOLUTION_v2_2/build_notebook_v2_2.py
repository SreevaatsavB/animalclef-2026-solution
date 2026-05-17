#!/usr/bin/env python3
"""
build_notebook_v2_2.py
======================
V2.2: RootSIFT descriptors + BFMatcher (NO LNBNN, NO MegaDescriptor).

Diagnostic purpose:
  V2.0 (0.10691) and V2.1 (0.10957) both added RootSIFT + LNBNN and
  collapsed catastrophically.  Root cause: LNBNN background distance
  collapses to near-zero in a 100k-descriptor pool → scores ≈ 0 →
  ensemble scale drops → clustering thresholds miscalibrated → every
  image is its own cluster.

  V2.2 keeps RootSIFT but drops LNBNN entirely, retaining the BFMatcher
  ratio test from V1.3.  This preserves ensemble score scale so clustering
  thresholds remain valid.

Changes from V1.3:
  1. RootSIFT — L1-norm + element-wise sqrt applied to SIFT descriptors
     immediately after detectAndCompute (before keypoint mask filter).
     Identical to the patch applied in V2.1.

That is the ONLY change.  Cell 4.2 (BFMatcher) is left completely
untouched so the ensemble output stays on the same scale as V1.3.

Expected outcomes:
  > 0.32528  → RootSIFT improves BFMatcher discrimination; LNBNN was
               the sole structural problem in V2.0/V2.1.
  ≈ 0.32528  → RootSIFT is neutral for BFMatcher; LNBNN was the issue.
  < 0.32528  → RootSIFT also hurts BFMatcher ratio test; both are bad.

Source: V1.3 notebook (not V2.0 or V2.1 — clean base).

Usage:
    python3 build_notebook_v2_2.py
"""

import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_NB  = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v1_3/ensemble_global_local_reid_v1_3.ipynb')
OUT_DIR = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v2_2')
OUT_NB  = OUT_DIR / 'ensemble_global_local_reid_v2_2.ipynb'

# ── RootSIFT patch (Cell 3.1) ──────────────────────────────────────────────────
# Insert RootSIFT transform immediately after the early-return guard and
# before the existing keypoint mask filter block.
# Marker strings match the V1.3 SIFT extractor exactly.

_ROOTSIFT_OLD = (
    '        if descriptors is None or len(keypoints) < 4:\n'
    '            return None\n'
    '        \n'
    '        # Filter keypoints to animal region using segmentation mask.'
)

_ROOTSIFT_NEW = (
    '        if descriptors is None or len(keypoints) < 4:\n'
    '            return None\n'
    '        \n'
    '        # RootSIFT: L1-normalize then element-wise sqrt (HotSpotter\'s descriptor).\n'
    '        # More discriminative for texture-heavy patterns (turtles, lizards).\n'
    '        # SIFT descriptors are non-negative (gradient-magnitude histograms).\n'
    '        descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-7)\n'
    '        descriptors = np.sqrt(descriptors)\n'
    '        \n'
    '        # Filter keypoints to animal region using segmentation mask.'
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def patch_cell_source(cells: list, old_str: str, new_str: str) -> bool:
    for cell in cells:
        src = ''.join(cell['source'])
        if old_str in src:
            cell['source'] = src.replace(old_str, new_str, 1)
            return True
    return False


def find_cell_index(cells: list, marker: str) -> int:
    for i, cell in enumerate(cells):
        if marker in ''.join(cell['source']):
            return i
    return -1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'Reading source notebook: {SRC_NB}')
    with open(SRC_NB) as f:
        nb = json.load(f)

    cells = nb['cells']
    n_src = len(cells)
    print(f'  Source cells: {n_src}')

    # ── 1. RootSIFT patch ────────────────────────────────────────────────────
    ok = patch_cell_source(cells, _ROOTSIFT_OLD, _ROOTSIFT_NEW)
    print(f'  RootSIFT patch {"OK" if ok else "FAILED — marker not found!"}')

    # ── 2. Kernel metadata ────────────────────────────────────────────────────
    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    print(f'  Total cells: {len(cells)} (unchanged from source — no insertions)')

    # ── 3. Write notebook ─────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_NB, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'\nOutput notebook: {OUT_NB}')

    # ── 4. Kaggle kernel-metadata.json ────────────────────────────────────────
    km = {
        'id': 'sreevaatsavbavana/animalclef-26-v2-2',
        'title': 'AnimalCLEF 26 V2.2 (RootSIFT + BFMatcher, no LNBNN)',
        'code_file': 'ensemble_global_local_reid_v2_2.ipynb',
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

    # ── 5. Verify ─────────────────────────────────────────────────────────────
    print('\n── Verification ──────────────────────────────────────────────────────')
    with open(OUT_NB) as f:
        nb_check = json.load(f)
    all_src = '\n'.join(''.join(c['source']) for c in nb_check['cells'])

    checks = [
        # ── V1.3 baseline (must still be present) ──────────────────────────
        ('Cell 1.6 present',            'Cell 1.6: SAM 3'              in all_src),
        ('SEG_MAP built',               'SEG_MAP built'                 in all_src),
        ('get_seg_mask defined',        'def get_seg_mask'              in all_src),
        ('ROOT_DIR str-safe',           'Path(ROOT_DIR)'                in all_src),
        ('Keypoint filter present',     'seg_mask = get_seg_mask'       in all_src),
        ('Filter keeps >=4 kpts',       'len(keep) >= 4'                in all_src),
        # ── V2.2: RootSIFT ─────────────────────────────────────────────────
        ('RootSIFT L1-norm',            'descriptors.sum(axis=1'        in all_src),
        ('RootSIFT sqrt',               'np.sqrt(descriptors)'          in all_src),
        # ── Must NOT contain LNBNN ─────────────────────────────────────────
        ('No LNBNN function',           'lnbnn_match_scores' not        in all_src),
        ('No scatter_add_',             'scatter_add_'       not        in all_src),
        ('No argpartition',             'argpartition'       not        in all_src),
        # ── Must NOT contain MegaDescriptor ────────────────────────────────
        ('No MegaDescriptor',           'MegaDescriptor'     not        in all_src),
        ('No timm.create_model',        'timm.create_model'  not        in all_src),
        # ── BFMatcher must still be present ────────────────────────────────
        ('BFMatcher present',           'BFMatcher'                     in all_src),
        # ── Cell count unchanged ────────────────────────────────────────────
        ('Cell count same as V1.3',     len(nb_check['cells']) == n_src),
    ]

    all_ok = True
    for name, result in checks:
        status = '✓' if result else '✗'
        if not result:
            all_ok = False
        print(f'  {status} {name}')

    print()
    print('All checks passed. V2.2 notebook is ready.' if all_ok
          else 'Some checks FAILED — review output above.')


if __name__ == '__main__':
    main()
