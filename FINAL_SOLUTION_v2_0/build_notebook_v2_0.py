#!/usr/bin/env python3
"""
build_notebook_v2_0.py
======================
V2.0: MegaDescriptor ensemble + RootSIFT + LNBNN scoring.

Changes from V1.3 (best score: 0.32528):

  1. Cells 2.3b/2.3c/2.3d — MegaDescriptor-L-384 as second global model
       - Wildlife re-ID backbone trained on 37k+ individuals (distinct from MiewID)
       - L2-normalized, then concatenated with MiewID and re-normalized: 3688-dim
       - The rest of the pipeline (QE, cosine sim, ensemble) works unchanged

  2. Cell 3.1 patch — RootSIFT descriptor upgrade
       - After detectAndCompute: L1-normalize then element-wise sqrt
       - More discriminative for texture-heavy patterns (turtles, lizards)
       - Two lines of code, no new packages

  3. Cell 4.2 replacement — LNBNN scoring (HotSpotter-style)
       - Replaces BFMatcher ratio test with Local Naive Bayes NN scoring
       - Each query descriptor votes for the db image that beats its background
       - Still uses top-100 global candidate pre-selection (runtime manageable)
       - Pure numpy/torch; no new packages

Usage:
    python3 build_notebook_v2_0.py
"""

import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_NB  = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v1_3/ensemble_global_local_reid_v1_3.ipynb')
OUT_DIR = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26/FINAL_SOLUTION_v2_0')
OUT_NB  = OUT_DIR / 'ensemble_global_local_reid_v2_0.ipynb'

# ── New Cell 2.3b: MegaDescriptor model loader ────────────────────────────────
CELL_2_3B_SOURCE = """\
# Cell 2.3b: MegaDescriptor-L-384 Model
#
# Wildlife re-ID backbone trained on 37k+ individuals across 247 datasets.
# Distinct visual cues from MiewID → complementary representation.
# Loaded via timm from HuggingFace Hub (requires internet or pre-cached weights).

def get_mega_model():
    \"\"\"Load MegaDescriptor-L-384 via timm (HuggingFace Hub).\"\"\"
    # num_classes=0 strips the classification head → returns 1536-dim feature embedding.
    # Without it, timm returns class logits (wrong dimension, wrong semantics).
    model = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0)
    model = model.eval().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model

print("✓ MegaDescriptor-L-384 model loader defined")
"""

# ── New Cell 2.3c: Extract + cache MegaDescriptor features per species ────────
CELL_2_3C_SOURCE = """\
# Cell 2.3c: Extract and Cache MegaDescriptor Features

# Swin-L at 384×384 does TWO forward passes per batch (TTA).  Use a smaller
# batch than the global BATCH_SIZE (64) to avoid OOM on P100 (16 GB).
# Increase to 64 if GPU memory allows; decrease to 16 if you see OOM errors.
MEGA_BATCH_SIZE = 32

def extract_mega_features(model, dataset, image_size=384):
    \"\"\"Extract L2-normalized MegaDescriptor features with TTA (horizontal flip).\"\"\"
    dataset.transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    loader = DataLoader(
        dataset,
        batch_size=MEGA_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting MegaDescriptor features", leave=False):
            images = batch[0].to(DEVICE)

            # TTA: original + horizontal flip
            with torch.cuda.amp.autocast():
                feats = model(images) + model(torch.flip(images, dims=[3]))

            feats_norm = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
            all_features.append(feats_norm.cpu().numpy())

    return np.concatenate(all_features)


mega_features_cache = {}
mega_model = get_mega_model()

for species in test_meta["dataset"].unique():
    print(f"\\n{'='*60}")
    print(f"Processing {species} - MegaDescriptor Features")

    cache_file = f"cache/embeddings/{species}_mega.npy"

    if os.path.exists(cache_file):
        print(f"  Loading cached embeddings: {cache_file}")
        features = np.load(cache_file)
    else:
        sp_meta = test_meta[test_meta["dataset"] == species]
        sp_dataset = full_dataset.get_subset(sp_meta.index.values)

        features = extract_mega_features(mega_model, sp_dataset, image_size=384)

        np.save(cache_file, features)
        print(f"  Cached to {cache_file}")

    mega_features_cache[species] = features
    print(f"  Shape: {features.shape}, Norm: {np.linalg.norm(features[0]):.3f}")

del mega_model
torch.cuda.empty_cache()
print("\\n✓ MegaDescriptor features extracted and cached")
"""

# ── New Cell 2.3d: Fuse MiewID + MegaDescriptor ───────────────────────────────
CELL_2_3D_SOURCE = """\
# Cell 2.3d: Fuse MiewID + MegaDescriptor Embeddings
#
# Both are already L2-normalized per species.
# MiewID (2152-dim) ++ MegaDescriptor (1536-dim) → concat → re-normalize → 3688-dim.
# global_features_cache is updated in-place; downstream cells (QE, cosine sim,
# top-K selection) are unchanged — they only consume global_features_cache.

for species in list(global_features_cache.keys()):
    miew  = global_features_cache[species]          # (N, 2152) L2-normalized
    mega  = mega_features_cache[species]             # (N, 1536) L2-normalized
    fused = np.concatenate([miew, mega], axis=1)    # (N, 3688)
    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)
    global_features_cache[species] = fused
    print(f"  {species}: fused shape {fused.shape}, norm={np.linalg.norm(fused[0]):.3f}")

print("\\n✓ Global features fused (MiewID + MegaDescriptor → 3688-dim)")
"""

# ── RootSIFT patch: two lines added after early-return, before mask filter ─────
# Exact strings from V1.3 Cell 3.1 (after V1.3's keypoint mask patch was applied)
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

# ── Cell 4.2: LNBNN full replacement ──────────────────────────────────────────
CELL_4_2_SOURCE = """\
# Cell 4.2: LNBNN Matching (HotSpotter-style scoring)
#
# Replaces BFMatcher + Lowe's ratio test with Local Naive Bayes Nearest Neighbor.
#
# Key idea: for each query descriptor q, find k+1 nearest neighbors across the
# full candidate database.  The (k+1)th distance is the "background" reference.
# Image i accumulates: sum(bg_dist - match_dist) for each query kpt whose
# k nearest neighbors land in image i.  This naturally weights images that
# contain many close (distinctive) matches rather than just counting them.
#
# Still uses top-100 global cosine-similarity pre-selection to bound runtime.
#
# GPU optimisations:
#   - db and index tensors uploaded once per query (H2D); topk done on GPU
#   - Accumulation via scatter_add_ stays on GPU — no D2H of topk results
#   - Match-matrix update is vectorised numpy (no Python inner loop)
#   - argpartition (O(N)) replaces argsort (O(N log N)) for top-K selection
#   - astype(..., copy=False) skips copies when descriptors already float32

def lnbnn_match_scores(query_descs, db_descs_list, k=3):
    \"\"\"
    LNBNN scoring: match one query image against a list of db images.

    Args:
        query_descs  : (Nq, D) float32 array (RootSIFT descriptors for query).
        db_descs_list: list of length n_db; each entry is an (Ni, D) float32
                       array or None (treated as missing → score stays 0).
        k            : number of nearest-neighbor votes per query descriptor.

    Returns:
        scores: (n_db,) float64 numpy array of non-negative LNBNN scores.
    \"\"\"
    if query_descs is None or len(query_descs) < 4:
        return np.zeros(len(db_descs_list))

    # Collect valid (non-None, non-empty) db entries, preserve original indices
    valid = [
        (orig_i, d) for orig_i, d in enumerate(db_descs_list)
        if d is not None and len(d) >= 1
    ]
    if not valid:
        return np.zeros(len(db_descs_list))

    orig_indices, valid_descs = zip(*valid)

    # Flat database — copy=False avoids a redundant copy if already float32
    db_flat    = np.concatenate(valid_descs, axis=0).astype(np.float32, copy=False)
    # int64 required for scatter_add_ index tensor
    db_img_idx = np.concatenate(
        [np.full(len(d), oi, dtype=np.int64) for oi, d in valid], axis=0
    )  # (N_total,) maps each db descriptor row → original db_descs_list index

    q_np = query_descs.astype(np.float32, copy=False)

    # Upload db and index to GPU once (H2D); topk stays on GPU for scatter_add_
    db             = torch.from_numpy(db_flat).to(DEVICE)
    db_img_idx_gpu = torch.from_numpy(db_img_idx).to(DEVICE)   # int64 LongTensor
    kk             = min(k + 1, db.shape[0])
    n_votes        = kk - 1

    # Accumulation buffer lives on GPU — no D2H until the very end
    scores_gpu = torch.zeros(len(db_descs_list), dtype=torch.float64, device=DEVICE)

    # Chunked GPU kNN: caps peak memory to ~100 MB per chunk
    CHUNK = 256
    for s in range(0, len(q_np), CHUNK):
        q_chunk = torch.from_numpy(q_np[s:s + CHUNK]).to(DEVICE)
        d_chunk = torch.cdist(q_chunk, db, p=2)                  # (chunk, N_total)
        td, ti  = torch.topk(d_chunk, kk, dim=1, largest=False)
        del d_chunk, q_chunk

        bg = td[:, -1]                                            # (chunk,) background dist

        # GPU scatter_add_ — no need to copy topk results back to CPU
        for ki in range(n_votes):
            img_ids = db_img_idx_gpu[ti[:, ki]]                   # (chunk,) long
            deltas  = (bg - td[:, ki]).double()                   # (chunk,) float64
            pos     = deltas > 0
            scores_gpu.scatter_add_(0, img_ids[pos], deltas[pos])

        del td, ti

    del db, db_img_idx_gpu
    return scores_gpu.cpu().numpy()                               # single D2H: K floats


def compute_pairwise_matches_fast(features_list, extractor_type, species):
    \"\"\"Pairwise LNBNN matching with top-K global candidate pre-selection.\"\"\"
    n = len(features_list)
    cache_file = f"cache/match_scores/{species}_{extractor_type}_matches.npy"

    if os.path.exists(cache_file):
        print(f"  ✓ Loading cached {extractor_type} match scores: {cache_file}")
        return np.load(cache_file)

    print(f"  Computing {extractor_type} LNBNN matches for {n} images...")

    K = min(100, n)

    global_feats = global_features_expanded[species]
    global_sim   = np.dot(global_feats, global_feats.T)

    # argpartition is O(N) vs argsort O(N log N); top-K order doesn't matter here
    top_k_all = np.argpartition(global_sim, -K, axis=1)[:, -K:]  # (N, K)

    match_matrix = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(match_matrix, 1.0)

    for i in tqdm(range(n), desc=f"LNBNN {extractor_type}"):
        if features_list[i] is None:
            continue

        candidates = top_k_all[i]

        # Pass None for j==i so LNBNN never matches descriptors against themselves
        db_descs_list = [
            features_list[j]['descriptors'] if (features_list[j] is not None and j != i) else None
            for j in candidates
        ]

        raw_scores = lnbnn_match_scores(
            features_list[i]['descriptors'],
            db_descs_list,
        )

        # Vectorised match-matrix update — replaces the K-iteration Python loop
        valid_mask = candidates != i
        js         = candidates[valid_mask]
        scores_01  = (1.0 - np.exp(-raw_scores[valid_mask] / 20.0)).astype(np.float32)

        # np.maximum with assignment: 2 numpy calls instead of 100 Python iters
        match_matrix[i, js] = np.maximum(match_matrix[i, js], scores_01)
        match_matrix[js, i] = np.maximum(match_matrix[js, i], scores_01)

    np.save(cache_file, match_matrix)
    print(f"  ✓ Cached to {cache_file}")

    return match_matrix

print("✓ LNBNN matching defined (HotSpotter-style scoring)")
"""

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


def replace_cell_source(cells: list, marker: str, new_source: str) -> bool:
    idx = find_cell_index(cells, marker)
    if idx == -1:
        return False
    cells[idx]['source'] = new_source
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'Reading source notebook: {SRC_NB}')
    with open(SRC_NB) as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f'  Source cells: {len(cells)}')

    # ── 1. Insert Cells 2.3b / 2.3c / 2.3d after Cell 2.3 ────────────────────
    idx_2_3 = find_cell_index(cells, 'Cell 2.3: Cache Global Embeddings')
    if idx_2_3 == -1:
        raise RuntimeError('Cell 2.3 not found in source notebook')
    # Insert in reverse order so each goes to idx_2_3+1 → final order is b,c,d
    cells.insert(idx_2_3 + 1, make_code_cell(CELL_2_3D_SOURCE))
    cells.insert(idx_2_3 + 1, make_code_cell(CELL_2_3C_SOURCE))
    cells.insert(idx_2_3 + 1, make_code_cell(CELL_2_3B_SOURCE))
    print(f'  Inserted Cells 2.3b/2.3c/2.3d after index {idx_2_3}')

    # ── 2. RootSIFT patch (Cell 3.1) ─────────────────────────────────────────
    ok = patch_cell_source(cells, _ROOTSIFT_OLD, _ROOTSIFT_NEW)
    print(f'  RootSIFT patch {"OK" if ok else "FAILED — marker not found!"}')

    # ── 3. Replace Cell 4.2 with LNBNN version ────────────────────────────────
    ok = replace_cell_source(cells, 'Cell 4.2: Ultra-Fast GPU Batch Matching', CELL_4_2_SOURCE)
    print(f'  Cell 4.2 LNBNN replacement {"OK" if ok else "FAILED — marker not found!"}')

    # ── 4. Kernel metadata ────────────────────────────────────────────────────
    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    print(f'  Total cells: {len(cells)}')

    # ── 5. Write notebook ─────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_NB, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'\nOutput notebook: {OUT_NB}')

    # ── 6. Kaggle kernel-metadata.json ────────────────────────────────────────
    km = {
        'id': 'sreevaatsavbavana/animalclef-26-v2-0',
        'title': 'AnimalCLEF 26 V2.0 (MegaDescriptor + RootSIFT + LNBNN)',
        'code_file': 'ensemble_global_local_reid_v2_0.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': True,   # Required for MegaDescriptor-L-384 (HF Hub)
        'dataset_sources': ['sreevaatsavbavana/animalclef-26-sam3'],
        'competition_sources': ['animal-clef-2026'],
        'kernel_sources': [],
        'model_sources': [],
    }
    km_path = OUT_DIR / 'kernel-metadata.json'
    with open(km_path, 'w') as f:
        json.dump(km, f, indent=2)
    print(f'Kaggle kernel-metadata:  {km_path}')

    # ── 7. Verify ─────────────────────────────────────────────────────────────
    print('\n── Verification ──────────────────────────────────────────────────────')
    with open(OUT_NB) as f:
        nb_check = json.load(f)
    all_src = '\n'.join(''.join(c['source']) for c in nb_check['cells'])

    checks = [
        # ── V1.3 baseline (must still be present) ──────────────────────────
        ('Cell 1.6 present',              'Cell 1.6: SAM 3'                  in all_src),
        ('SEG_MAP built',                 'SEG_MAP built'                     in all_src),
        ('get_seg_mask defined',          'def get_seg_mask'                  in all_src),
        ('ROOT_DIR str-safe',             'Path(ROOT_DIR)'                    in all_src),
        ('Keypoint filter present',       'seg_mask = get_seg_mask'           in all_src),
        ('Filter keeps >=4 kpts',         'len(keep) >= 4'                    in all_src),
        ('SEG_CACHE_ROOT set',            'SEG_CACHE_ROOT'                    in all_src),
        # ── V2.0: MegaDescriptor ───────────────────────────────────────────
        ('MegaDescriptor loader',         'def get_mega_model'                in all_src),
        ('MegaDescriptor model name',     'MegaDescriptor-L-384'              in all_src),
        ('num_classes=0 present',         'num_classes=0'                     in all_src),
        ('mega_features_cache built',     'mega_features_cache = {}'          in all_src),
        ('Feature fusion cell',           'Global features fused'             in all_src),
        ('Fused concat present',          'np.concatenate([miew, mega]'       in all_src),
        # ── V2.0: RootSIFT ─────────────────────────────────────────────────
        ('RootSIFT L1-norm',              'descriptors.sum(axis=1'            in all_src),
        ('RootSIFT sqrt',                 'np.sqrt(descriptors)'              in all_src),
        # ── V2.0: LNBNN ────────────────────────────────────────────────────
        ('LNBNN function defined',        'def lnbnn_match_scores'            in all_src),
        ('LNBNN bg_dist',                 'bg_dist'                           in all_src),
        ('LNBNN used in compute_pairwise','lnbnn_match_scores'                in all_src),
        ('No old batch_sift_match_gpu',   'batch_sift_match_gpu' not          in all_src),
        ('GPU scatter_add accumulation',  'scatter_add_'                      in all_src),
        ('copy=False on astype',          'copy=False'                        in all_src),
        ('argpartition not argsort',      'argpartition'                      in all_src),
        ('Vectorised matrix update',      'np.maximum(match_matrix'           in all_src),
        ('Chunked cdist present',         'CHUNK'                             in all_src),
        ('Self excluded from LNBNN db',   'j != i'                            in all_src),
        ('Separate MEGA_BATCH_SIZE',      'MEGA_BATCH_SIZE'                   in all_src),
        # ── Cell counts ────────────────────────────────────────────────────
        ('Cell count increased by 3',     len(nb_check['cells']) == len(nb['cells'])),
    ]

    all_ok = True
    for name, result in checks:
        status = '✓' if result else '✗'
        if not result:
            all_ok = False
        print(f'  {status} {name}')

    print()
    print('All checks passed. V2.0 notebook is ready.' if all_ok
          else 'Some checks FAILED — review output above.')


if __name__ == '__main__':
    main()
