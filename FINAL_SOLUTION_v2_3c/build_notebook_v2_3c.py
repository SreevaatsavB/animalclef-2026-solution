#!/usr/bin/env python3
"""
build_notebook_v2_3c.py
=======================
V2.3c: Add KAZE local features to TexasHornedLizards (in addition to V2.3b's
KAZE for SeaTurtleID2022).

Background:
  TexasHornedLizards shares the same properties as SeaTurtleID2022 that made
  KAZE beneficial:
    - Rigid body → keypoints are stable across views
    - 100% SAM3 cache coverage → every image gets mask-filtered keypoints
    - Dense surface patterns (spots/scales) → non-linear scale space finds
      different stable keypoints than SIFT's Gaussian pyramid

Changes from V2.3b:
  1. TexasHornedLizards SPECIES_CONFIG:
       local_extractors: ["sift"] → ["sift", "kaze"]
       local_weights:    {"sift": 0.35} → {"sift": 0.20, "kaze": 0.15}
       global_weight stays 0.65; total = 0.65+0.20+0.15 = 1.00 ✓

  KAZEExtractor class and get_extractor 'kaze' branch already exist in V2.3b.
  No new code needed.

Source: V2.3b notebook (already has KAZE infrastructure for SeaTurtle).

Usage:
    python3 build_notebook_v2_3c.py
"""

import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_NB  = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v2_3b/ensemble_global_local_reid_v2_3b.ipynb')
OUT_DIR = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v2_3c')
OUT_NB  = OUT_DIR / 'ensemble_global_local_reid_v2_3c.ipynb'

# ── Patch: TexasHornedLizards SPECIES_CONFIG — add KAZE ───────────────────────
_TEXAS_OLD = (
    '    "TexasHornedLizards": {\n'
    '        # Dense spot patterns \u2192 SIFT\n'
    '        "global_weight": 0.65,\n'
    '        "local_extractors": ["sift"],\n'
    '        "local_weights": {"sift": 0.35},'
)

_TEXAS_NEW = (
    '    "TexasHornedLizards": {\n'
    '        # Dense spot patterns \u2192 SIFT + KAZE (non-linear scale space)\n'
    '        "global_weight": 0.65,\n'
    '        "local_extractors": ["sift", "kaze"],\n'
    '        "local_weights": {"sift": 0.20, "kaze": 0.15},'
)

# ── Helpers ────────────────────────────────────────────────────────────────────

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
    n_src = len(cells)
    print(f'  Source cells: {n_src}')

    # Patch TexasHornedLizards config
    ok = patch_cell_source(cells, _TEXAS_OLD, _TEXAS_NEW)
    print(f'  TexasHornedLizards config patch {"OK" if ok else "FAILED — marker not found!"}')

    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    print(f'  Total cells: {len(cells)} (unchanged from source)')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_NB, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'\nOutput notebook: {OUT_NB}')

    km = {
        'id': 'sreevaatsavbavana/animalclef-26-v2-3c',
        'title': 'AnimalCLEF 26 V2.3c (KAZE SeaTurtle + TexasLizard)',
        'code_file': 'ensemble_global_local_reid_v2_3c.ipynb',
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

    # ── Verification ──────────────────────────────────────────────────────────
    print('\n── Verification ──────────────────────────────────────────────────────')
    with open(OUT_NB) as f:
        nb_check = json.load(f)
    all_src = '\n'.join(''.join(c['source']) for c in nb_check['cells'])

    checks = [
        # V2.2 baseline preserved
        ('RootSIFT on SIFT present',            'descriptors.sum(axis=1'                in all_src),
        ('SAM3 mask filter present',            'seg_mask = get_seg_mask'               in all_src),
        ('BFMatcher present',                   'BFMatcher'                             in all_src),
        ('No LNBNN',                            'lnbnn_match_scores' not                in all_src),
        # KAZE infrastructure (inherited from V2.3b)
        ('KAZEExtractor defined',               'class KAZEExtractor'                   in all_src),
        ('cv2.KAZE_create used',                'cv2.KAZE_create'                       in all_src),
        ('KAZE no RootSIFT note',               'sum_dx' in all_src and 'NaN'           in all_src),
        ('kaze branch in get_extractor',        "elif extractor_name == 'kaze'"         in all_src),
        # SeaTurtle unchanged from V2.3b
        ('SeaTurtle sift=0.20',                 '"sift": 0.20'                          in all_src),
        ('SeaTurtle kaze=0.15',                 '"kaze": 0.15'                          in all_src),
        # TexasHornedLizards now has KAZE
        ('Texas extractors has kaze',           '"local_extractors": ["sift", "kaze"]'  in all_src),
        ('Texas sift weight=0.20',              # both SeaTurtle and Texas use 0.20
                                                all_src.count('"sift": 0.20') >= 2),
        ('Texas kaze weight=0.15',
                                                all_src.count('"kaze": 0.15') >= 2),
        # Other species unchanged
        ('Salamander sift-only',                '"SalamanderID2025"'                    in all_src),
        ('Lynx sift-only',                      '"LynxID2025"'                          in all_src),
        # No kornia active code paths
        ('No aliked route in get_extractor',    "elif extractor_name == 'aliked'" not   in all_src),
        ('No KF.DISK active',                   'KF.DISK.from_pretrained' not           in all_src),
        # Cell count unchanged
        ('Cell count same as V2.3b',            len(nb_check['cells']) == n_src),
    ]

    all_ok = True
    for name, result in checks:
        status = '✓' if result else '✗'
        if not result:
            all_ok = False
        print(f'  {status} {name}')

    print()
    print('All checks passed. V2.3c notebook is ready.' if all_ok
          else 'Some checks FAILED — review output above.')


if __name__ == '__main__':
    main()
