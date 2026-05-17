#!/usr/bin/env python3
"""
build_notebook_v2_5.py
======================
V2.5: KAZE for SalamanderID2025 + LynxID2025, plus per-species threshold
calibration from training identity labels.

Changes from V2.3c:
  1. SalamanderID2025:
       local_extractors: ["sift"] → ["sift", "kaze"]
       global_weight:    0.70 → 0.65
       local_weights:    {"sift": 0.30} → {"sift": 0.20, "kaze": 0.15}
       SAM3 coverage 100% → KAZE keypoints are mask-filtered (same as SeaTurtle/THL)

  2. LynxID2025:
       local_extractors: ["sift"] → ["sift", "kaze"]
       global_weight:    0.70 → 0.70  (kept higher — 0% SAM3 coverage)
       local_weights:    {"sift": 0.30} → {"sift": 0.20, "kaze": 0.10}
       NOTE: Lynx has 0% SAM3 coverage → KAZE runs on original images.
             Smaller KAZE weight (0.10) to limit background-keypoint noise.

  3. New cell (inserted after Cell 5 "Device Detection"):
       Copies precomputed KAZE local features + match scores from the
       sreevaatsavbavana/v2-5-cache Kaggle dataset into the working cache/
       directory so the existing cache-check logic loads them instead of
       re-extracting.

  4. New cell (inserted after Cell 28 "Weighted Voting Summary"):
       Threshold calibration via training identity labels.
       For each of Lynx / Salamander / SeaTurtle (all have train splits):
         - Reload MiewID model, extract global embeddings for train images
         - Grid-search threshold in [0.15, 0.60] to maximise AMI
       THL: no training split → keep V2.3c threshold (0.30).
       Calibrated thresholds are written back into SPECIES_CONFIG so the
       existing clustering cell picks them up automatically.

Source: FINAL_SOLUTION_v2_3c/ensemble_global_local_reid_v2_3c.ipynb

Usage:
    python3 build_notebook_v2_5.py
"""

import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_NB  = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v2_3c/ensemble_global_local_reid_v2_3c.ipynb')
OUT_DIR = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v2_5')
OUT_NB  = OUT_DIR / 'ensemble_global_local_reid_v2_5.ipynb'

# ── Patch 1: SalamanderID2025 — add KAZE ─────────────────────────────────────
_SALAMANDER_OLD = (
    '    "SalamanderID2025": {\n'
    '        # Deformable bodies \u2192 SIFT only (rotation invariant)\n'
    '        "global_weight": 0.70,\n'
    '        "local_extractors": ["sift"],\n'
    '        "local_weights": {"sift": 0.30},'
)

_SALAMANDER_NEW = (
    '    "SalamanderID2025": {\n'
    '        # Deformable bodies \u2192 SIFT + KAZE (SAM3 100% coverage \u2192 masked keypoints)\n'
    '        "global_weight": 0.65,\n'
    '        "local_extractors": ["sift", "kaze"],\n'
    '        "local_weights": {"sift": 0.20, "kaze": 0.15},'
)

# ── Patch 2: LynxID2025 — add KAZE (conservative; 0% SAM3) ───────────────────
_LYNX_OLD = (
    '    "LynxID2025": {\n'
    '        # Rotation invariance \u2192 SIFT (perfect for this)\n'
    '        "global_weight": 0.70,\n'
    '        "local_extractors": ["sift"],\n'
    '        "local_weights": {"sift": 0.30},'
)

_LYNX_NEW = (
    '    "LynxID2025": {\n'
    '        # Rosette/fur patterns \u2192 SIFT + KAZE (0% SAM3 \u2192 on originals; conservative weight)\n'
    '        "global_weight": 0.70,\n'
    '        "local_extractors": ["sift", "kaze"],\n'
    '        "local_weights": {"sift": 0.20, "kaze": 0.10},'
)

# ── New cell 1: Copy precomputed KAZE cache from Kaggle dataset ───────────────
# Inserted right after Cell 5 (device detection + cache dir creation).
# Uses shutil.copy2 so existing cache entries are never overwritten.
_V25_CACHE_CELL_SOURCE = '''\
# V2.5 Cell 1.4b: Preload KAZE cache from sreevaatsavbavana/v2-5-cache dataset
# ─────────────────────────────────────────────────────────────────────────────
# The v2-5-cache dataset contains precomputed KAZE local features and match
# scores for SalamanderID2025 and LynxID2025.  Copying them here means the
# existing cache-check logic in Cells 3.5 and 4.4 will load them directly
# without re-extracting.

import shutil

# Kaggle mounts datasets at /kaggle/input/{slug}/; try both naming conventions
_V25_CANDIDATES = [
    "/kaggle/input/datasets/sreevaatsavbavana/v2-5-cache/cache",
    "/kaggle/input/v2-5-cache/cache",
    "/kaggle/input/v2-5-cache/cache",
]

_V25_SRC = None
for _p in _V25_CANDIDATES:
    if os.path.exists(_p):
        _V25_SRC = _p
        break

if _V25_SRC:
    print(f"V2.5 cache found: {_V25_SRC}")
    _copied = 0
    for _sub in ["local_features", "match_scores"]:
        _src_dir = os.path.join(_V25_SRC, _sub)
        _dst_dir = os.path.join("cache", _sub)
        if os.path.isdir(_src_dir):
            for _fname in sorted(os.listdir(_src_dir)):
                _src_f = os.path.join(_src_dir, _fname)
                _dst_f = os.path.join(_dst_dir, _fname)
                if not os.path.exists(_dst_f):
                    shutil.copy2(_src_f, _dst_f)
                    print(f"  Copied  {_sub}/{_fname}")
                    _copied += 1
                else:
                    print(f"  Present {_sub}/{_fname}")
    print(f"✓ V2.5 cache preloaded ({_copied} new files)")
else:
    print("⚠ V2.5 cache dataset not mounted — KAZE features will be extracted from scratch")
'''

# ── New cell 2: Threshold calibration using training identity labels ──────────
# Inserted after Cell 28 "Weighted Voting Summary", before Section 6.
# Uses single-quote docstrings to avoid JSON/cell-source escape issues.
_CALIB_CELL_SOURCE = """\
# Cell 5.4: Threshold Calibration using Training Identity Labels
# ──────────────────────────────────────────────────────────────
# For species with training splits (Lynx, Salamander, SeaTurtle) we grid-search
# the clustering threshold that maximises AMI against known identities.
# Uses global cosine similarity only (fast; global weight dominates 0.65-0.70).
# THL has no training split -> keeps V2.3c threshold (0.30).
#
# Calibrated thresholds are written back into SPECIES_CONFIG so Cell 6.2 picks
# them up automatically -- no other cells need changing.

import time as _time
from PIL import Image as _PILImg
from sklearn.metrics import adjusted_mutual_info_score as _ami_score

CALIB_SPECIES   = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
CALIB_MAX_IMGS  = 2500   # cap for SeaTurtle (8729 train imgs -> subsample)
CALIB_THR_STEPS = 46     # 0.15 -> 0.60 step 0.01

print("=" * 60)
print("V2.5 -- Threshold Calibration (Training Identities)")
print("=" * 60)

def _extract_calib_embeddings(model, image_paths, image_size, batch_size=64):
    '''Extract L2-normalised embeddings from a list of image paths.'''
    _tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    all_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = _PILImg.open(p).convert("RGB")
                    imgs.append(_tfm(img))
                except Exception:
                    imgs.append(torch.zeros(3, image_size, image_size))
            batch = torch.stack(imgs).to(DEVICE)
            with torch.cuda.amp.autocast():
                feats = model(batch) + model(torch.flip(batch, dims=[3]))
            feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats).astype(np.float32)

# Reload MiewID (was deleted after Cell 2.3 to free VRAM)
print("\\nLoading MiewID for calibration...")
calib_model = get_global_model()
calib_model.eval()

calibrated_thresholds = {}

for sp in CALIB_SPECIES:
    t0 = _time.time()
    sp_train = train_meta[train_meta["dataset"] == sp].copy()

    # Subsample large species to stay within memory/time budget
    if len(sp_train) > CALIB_MAX_IMGS:
        sp_train = sp_train.sample(n=CALIB_MAX_IMGS, random_state=42).sort_values("image_id")
        print(f"\\n{sp}: subsampled {len(sp_train):,} / {len(train_meta[train_meta['dataset'] == sp]):,} train images")
    else:
        print(f"\\n{sp}: {len(sp_train):,} train images")

    true_labels, _ = pd.factorize(sp_train["identity"].values)
    print(f"  identities: {len(np.unique(true_labels)):,}")

    # Build absolute image paths
    img_paths = [os.path.join(ROOT_DIR, p) for p in sp_train["path"].values]

    # Extract global embeddings (GPU, with TTA)
    cfg       = SPECIES_CONFIG[sp]
    train_emb = _extract_calib_embeddings(calib_model, img_paths, cfg["image_size"])

    # Cosine similarity matrix (global only -- fast, dominant weight)
    cos_sim = np.clip(train_emb @ train_emb.T, 0.0, 1.0).astype(np.float32)

    # Grid-search threshold [0.15, 0.60] step 0.01
    best_thr, best_ami = cfg["threshold_cluster"], -1.0
    for step in range(CALIB_THR_STEPS):
        thr  = round(0.15 + step * 0.01, 2)
        dist = np.clip(1.0 - cos_sim, 0.0, 1.0).astype(np.float64)
        pred = AgglomerativeClustering(
            n_clusters=None, metric="precomputed",
            linkage="average", distance_threshold=thr,
        ).fit_predict(dist)
        ami = _ami_score(true_labels, pred)
        if ami > best_ami:
            best_ami, best_thr = ami, thr

    calibrated_thresholds[sp] = best_thr
    prev = cfg["threshold_cluster"]
    print(f"  threshold: {prev:.2f} -> {best_thr:.2f}  (AMI={best_ami:.4f}, t={_time.time()-t0:.1f}s)")

del calib_model
torch.cuda.empty_cache()

# THL has no training split -- keep V2.3c threshold
calibrated_thresholds["TexasHornedLizards"] = SPECIES_CONFIG["TexasHornedLizards"]["threshold_cluster"]
print(f"\\nTexasHornedLizards: no train split -> threshold kept at {calibrated_thresholds['TexasHornedLizards']:.2f}")

# Write calibrated thresholds back into SPECIES_CONFIG (used by Cell 6.2)
for sp, thr in calibrated_thresholds.items():
    SPECIES_CONFIG[sp]["threshold_cluster"] = thr

print("\\nCalibrated SPECIES_CONFIG thresholds:")
for sp, cfg in SPECIES_CONFIG.items():
    print(f"  {sp:<25}: {cfg['threshold_cluster']:.2f}")
print("\\n✓ Thresholds updated -- Cell 6.2 will use these values")
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


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

    # ── Patch 1: SalamanderID2025 config ────────────────────────────────────
    ok1 = patch_cell_source(cells, _SALAMANDER_OLD, _SALAMANDER_NEW)
    print(f'  Salamander KAZE patch   {"OK" if ok1 else "FAILED -- marker not found!"}')

    # ── Patch 2: LynxID2025 config ──────────────────────────────────────────
    ok2 = patch_cell_source(cells, _LYNX_OLD, _LYNX_NEW)
    print(f'  Lynx KAZE patch         {"OK" if ok2 else "FAILED -- marker not found!"}')

    # ── Insert cache-loading cell after Cell 1.4 (device setup) ─────────────
    idx_device = find_cell_index(cells, 'Cell 1.4: Device Detection')
    if idx_device == -1:
        idx_device = find_cell_index(cells, 'ROOT_DIR')
    cache_insert_at = idx_device + 1
    cells.insert(cache_insert_at, make_code_cell(_V25_CACHE_CELL_SOURCE))
    print(f'  Cache-preload cell inserted at index {cache_insert_at}')

    # ── Insert calibration cell after Cell 5.3 (Weighted Voting Summary) ────
    # Note: indices shifted by 1 due to the cache cell insertion above
    idx_voting = find_cell_index(cells, 'Cell 5.3: Weighted Voting Summary')
    if idx_voting == -1:
        idx_voting = find_cell_index(cells, 'Section 6: Clustering')
        idx_voting = max(idx_voting - 1, len(cells) - 5)
    calib_insert_at = idx_voting + 1
    cells.insert(calib_insert_at, make_code_cell(_CALIB_CELL_SOURCE))
    print(f'  Calibration cell inserted at index {calib_insert_at}')

    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    print(f'  Total cells: {len(cells)} (= {n_src} + 2 new cells)')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_NB, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'\nOutput notebook: {OUT_NB}')

    # ── kernel-metadata.json ────────────────────────────────────────────────
    km = {
        'id': 'sreevaatsavbavana/animalclef-26-v2-5',
        'title': 'AnimalCLEF 26 V2.5 (KAZE all species + threshold calib)',
        'code_file': 'ensemble_global_local_reid_v2_5.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': [
            'sreevaatsavbavana/animalclef-26-sam3',
            'sreevaatsavbavana/v2-5-cache',
        ],
        'competition_sources': ['animal-clef-2026'],
        'kernel_sources': [],
        'model_sources': [],
    }
    km_path = OUT_DIR / 'kernel-metadata.json'
    with open(km_path, 'w') as f:
        json.dump(km, f, indent=2)
    print(f'Kaggle kernel-metadata:  {km_path}')

    # ── Verification ────────────────────────────────────────────────────────
    print('\n-- Verification ----------------------------------------------------------')
    with open(OUT_NB) as f:
        nb_check = json.load(f)
    all_src = '\n'.join(''.join(c['source']) for c in nb_check['cells'])

    checks = [
        # V2.3c baseline preserved
        ('RootSIFT present',                    'descriptors.sum(axis=1'                    in all_src),
        ('SAM3 mask filter present',            'seg_mask = get_seg_mask'                   in all_src),
        ('BFMatcher present',                   'BFMatcher'                                 in all_src),
        ('KAZEExtractor defined',               'class KAZEExtractor'                       in all_src),
        ('No LNBNN',                            'lnbnn_match_scores' not                    in all_src),
        # SeaTurtle + THL unchanged
        ('SeaTurtle sift=0.20',                 '"sift": 0.20'                              in all_src),
        ('SeaTurtle kaze=0.15',                 '"kaze": 0.15'                              in all_src),
        ('THL threshold=0.30',                  '"threshold_cluster": 0.30'                 in all_src),
        # Salamander now has KAZE
        ('Salamander has SAM3 comment',         'SAM3 100% coverage'                        in all_src),
        ('3 species have sift+kaze extractors',
                                                all_src.count('"local_extractors": ["sift", "kaze"]') >= 3),
        ('Salamander kaze=0.15 (3 occurrences)',
                                                all_src.count('"kaze": 0.15') >= 3),
        # Lynx now has KAZE
        ('Lynx has rosette comment',            'Rosette/fur patterns'                      in all_src),
        ('Lynx kaze=0.10',                      '"kaze": 0.10'                              in all_src),
        # Cache preload cell
        ('Cache preload cell present',          'v2-5-cache'                                in all_src),
        ('shutil.copy2 present',               'shutil.copy2'                               in all_src),
        # Calibration cell
        ('Calibration cell present',            'Threshold Calibration'                     in all_src),
        ('adjusted_mutual_info_score import',   'adjusted_mutual_info_score'                in all_src),
        ('CALIB_SPECIES defined',               'CALIB_SPECIES'                             in all_src),
        ('Single-quote docstring used',         "'''Extract L2"                             in all_src),
        ('No backslash-escaped docstring',      '\\"\\"\\"' not                             in all_src),
        ('calib writes back to SPECIES_CONFIG', 'threshold_cluster"] = thr'                 in all_src),
        ('THL kept in calibration',             'no train split'                            in all_src),
        # Cell count
        ('Cell count = source + 2',             len(nb_check['cells']) == n_src + 2),
        # Old strings gone
        ('Salamander OLD string gone',          _SALAMANDER_OLD                not          in all_src),
        ('Lynx OLD string gone',                _LYNX_OLD                      not          in all_src),
    ]

    all_ok = True
    for name, result in checks:
        status = 'OK' if result else 'FAIL'
        if not result:
            all_ok = False
        print(f'  [{status}] {name}')

    print()
    if all_ok:
        print('All checks passed. V2.5 notebook is ready.')
    else:
        print('Some checks FAILED -- review output above.')


if __name__ == '__main__':
    main()
