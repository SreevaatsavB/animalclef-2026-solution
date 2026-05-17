"""
Build script for V4.0 notebook.

RESULT: V4.0 = 0.4860 (new best, up from V3.2.2 = 0.47912)

Changes from V3.2:
  1. Add MegaDescriptor-L-384 as SEPARATE global backbone (1536-dim, not concatenated).
     V2.0 concatenated MiewID+MegaDesc into 3688-dim vector and failed (0.10691),
     but that was caused by LNBNN, not MegaDescriptor. Paper 251 (winner, 0.713)
     uses MegaDescriptor as a SEPARATE similarity matrix with its own weight.
     RESULT: MegaDescriptor helps SeaTurtle (mgw=0.30) but is useless for
     Lynx (mgw=0.00) and Salamander (mgw=0.00). Calibration correctly zeroes it.
  2. Split 'global_weight' into 'miew_weight' + 'mega_weight' in SPECIES_CONFIG.
  3. Keep AMI calibration metric. ARI was tried first (V4.0-ARI = 0.41769) but
     caused catastrophic Lynx over-merging (946 images → 8 clusters at thr=0.75).
     ARI rewards large clusters → pushes thresholds too high. AMI reverted.
  4. Bake in V3.2.2 fixes:
     - THR_GRID extended to [0.15..0.75] (31 steps, was 23).
     - KAZE explicit in grid (not residual via aw).
  5. Update calibration grid for dual-global (MIEW_W_GRID + MEGA_W_GRID).
     1260 valid weight combos × 31 thresholds per species.
  6. Update SPECIES_CONFIG defaults to V3.2.2 calibrated values.
  7. ROOT_DIR fallback for Kaggle path change.

Calibrated weights (V4.0, AMI):
  LynxID2025:        mw=0.40 mgw=0.00 sw=0.05 kw=0.00 aw=0.55 thr=0.67 → 36 clusters
  SalamanderID2025:  mw=0.40 mgw=0.00 sw=0.40 kw=0.15 aw=0.05 thr=0.51 → 289 clusters
  SeaTurtleID2022:   mw=0.50 mgw=0.30 sw=0.10 kw=0.10 aw=0.00 thr=0.65 → 112 clusters
  THL (uncalib):     mw=0.275 mgw=0.275 sw=0.15 kw=0.10 aw=0.20 thr=0.30 → 201 clusters
"""

import json, copy, pathlib, sys

SRC = pathlib.Path("FINAL_SOLUTION_v3_2/ensemble_global_local_reid_v3_2.ipynb")
DST = pathlib.Path("FINAL_SOLUTION_v4_0/ensemble_global_local_reid_v4_0.ipynb")

assert SRC.exists(), f"Source notebook not found: {SRC}"

with open(SRC) as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)

# --- Helpers -----------------------------------------------------------------

def patch_cell(nb, old, new, description):
    """Find + replace a unique substring across all cells."""
    count = 0
    for cell in nb["cells"]:
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        if old in src:
            count += 1
            src = src.replace(old, new, 1)
        cell["source"] = src
    assert count == 1, (
        f"Patch '{description}': expected 1 occurrence, found {count}\n"
        f"  looking for: {repr(old[:80])}"
    )


def replace_cell_source(nb, anchor, new_source, description):
    """Replace the ENTIRE source of the cell containing `anchor`."""
    count = 0
    for cell in nb["cells"]:
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        if anchor in src:
            count += 1
            cell["source"] = new_source
    assert count == 1, (
        f"Replace '{description}': expected 1 cell with anchor, found {count}\n"
        f"  looking for: {repr(anchor[:80])}"
    )


def insert_cell_after(nb, anchor, new_source, description, cell_type="code"):
    """Insert a new cell after the cell containing `anchor`."""
    for i, cell in enumerate(nb["cells"]):
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        if anchor in src:
            new_cell = {
                "cell_type": cell_type,
                "metadata": {},
                "source": new_source,
                "outputs": [],
                "execution_count": None,
            }
            nb["cells"].insert(i + 1, new_cell)
            return
    raise ValueError(
        f"Insert '{description}': anchor not found\n"
        f"  looking for: {repr(anchor[:80])}"
    )


# =============================================================================
# Patch 0: Version comment
# =============================================================================
patch_cell(
    nb,
    "V3.2: Fix calib/test score mismatch \u2014 ratio-test-only for SIFT/KAZE (no RANSAC)",
    "V4.0: MegaDescriptor-L dual-global + AMI calibration (Paper 251 architecture)",
    "version comment",
)


# =============================================================================
# Patch 1: SPECIES_CONFIG  (Cell 4)
# =============================================================================
# Split global_weight into miew_weight + mega_weight.
# Defaults are V3.2.2 calibrated values with gw split 50/50 as starting point.
# Calibration will find optimal split.

_NEW_SPECIES_CONFIG = """\
# Cell 1.3: Configuration - Species-Specific Weights

# V4.0: Dual-global (MiewID v3 + MegaDescriptor-L-384) + AMI calibration
# MegaDescriptor kept as SEPARATE similarity matrix (not concatenated like V2.0)
# Defaults: V3.2.2 calibrated values with gw split for miew/mega starting point

SPECIES_CONFIG = {
    "SalamanderID2025": {
        # Deformable bodies \u2014 SIFT + KAZE + ALIKED (SAM3 100% \u2192 masked keypoints)
        "miew_weight": 0.25,
        "mega_weight": 0.25,
        "local_extractors": ["sift", "kaze", "aliked"],
        "local_weights": {"sift": 0.30, "kaze": 0.15, "aliked": 0.05},
        "threshold_known": 0.40,
        "threshold_cluster": 0.47,
        "image_size": 512,
        "qe_k": 3,
    },
    "SeaTurtleID2022": {
        # Rigid, high-contrast features \u2014 SIFT + KAZE + ALIKED
        "miew_weight": 0.35,
        "mega_weight": 0.35,
        "local_extractors": ["sift", "kaze", "aliked"],
        "local_weights": {"sift": 0.20, "kaze": 0.05, "aliked": 0.05},
        "threshold_known": 0.45,
        "threshold_cluster": 0.57,
        "image_size": 512,
        "qe_k": 8,
    },
    "LynxID2025": {
        # Rosette/fur patterns \u2014 SIFT + KAZE + ALIKED (0% SAM3 \u2192 originals)
        "miew_weight": 0.20,
        "mega_weight": 0.20,
        "local_extractors": ["sift", "kaze", "aliked"],
        "local_weights": {"sift": 0.00, "kaze": 0.05, "aliked": 0.55},
        "threshold_known": 0.40,
        "threshold_cluster": 0.65,
        "image_size": 512,
        "qe_k": 5,
    },
    "TexasHornedLizards": {
        # Dense spot patterns \u2014 SIFT + KAZE + ALIKED
        "miew_weight": 0.275,
        "mega_weight": 0.275,
        "local_extractors": ["sift", "kaze", "aliked"],
        "local_weights": {"sift": 0.15, "kaze": 0.10, "aliked": 0.20},
        "threshold_known": None,  # Zero-shot
        "threshold_cluster": 0.30,
        "image_size": 512,
        "qe_k": 5,
    },
}

# Verify weights sum to 1.0
for species, cfg in SPECIES_CONFIG.items():
    total = cfg["miew_weight"] + cfg["mega_weight"] + sum(cfg["local_weights"].values())
    assert abs(total - 1.0) < 0.01, f"{species} weights don't sum to 1.0: {total}"

print("\u2713 Species configuration loaded (V4.0: dual-global)")
for species, cfg in SPECIES_CONFIG.items():
    extractors_str = " + ".join(
        f"{k.upper()}={v:.0%}" for k, v in cfg["local_weights"].items()
    )
    print(f"  {species}: MiewID {cfg['miew_weight']:.0%}, MegaDesc {cfg['mega_weight']:.0%}, {extractors_str}")
"""

replace_cell_source(
    nb,
    "# Cell 1.3: Configuration - Species-Specific Weights",
    _NEW_SPECIES_CONFIG,
    "SPECIES_CONFIG dual-global",
)


# =============================================================================
# Patch 2: Section 2 markdown header
# =============================================================================
patch_cell(
    nb,
    "## Section 2: Global Features (MiewID v3)",
    "## Section 2: Global Features (MiewID v3 + MegaDescriptor-L-384)",
    "section 2 header",
)


# =============================================================================
# Patch 3: Insert MegaDescriptor model + extraction + caching  (after Cell 12)
# =============================================================================
# Insert TWO new cells after the MiewID caching cell (Cell 12):
#   Cell 12a: MegaDescriptor model + extraction function
#   Cell 12b: Cache MegaDescriptor embeddings + query expansion

_MEGA_MODEL_CELL = """\
# Cell 2.3b: MegaDescriptor-L-384 Model + Extraction

# V4.0: MegaDescriptor as SEPARATE global backbone (not concatenated with MiewID).
# Paper 251 (1st place, 0.713) used MegaDescriptor-L + MiewID as dual globals.
# Key: num_classes=0 strips classification head \u2192 returns 1536-dim feature embedding.

MEGA_BATCH_SIZE = 32   # Swin-L at 384x384 + TTA = 2 fwds/batch; reduce to 16 if OOM
MEGA_IMAGE_SIZE = 384  # MegaDescriptor-L-384 native resolution


def get_mega_model():
    \"\"\"Load MegaDescriptor-L-384 via timm (HuggingFace Hub).\"\"\"
    model = timm.create_model(
        'hf-hub:BVRA/MegaDescriptor-L-384',
        pretrained=True,
        num_classes=0,  # CRITICAL: without this, returns class logits not embeddings
    )
    model = model.eval().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


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
            with torch.cuda.amp.autocast():
                feats = model(images) + model(torch.flip(images, dims=[3]))
            feats_norm = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
            all_features.append(feats_norm.cpu().numpy())

    return np.concatenate(all_features)


print("\u2713 MegaDescriptor-L-384 model + extraction function defined")
"""

_MEGA_CACHE_CELL = """\
# Cell 2.3c: Cache MegaDescriptor Embeddings + Query Expansion

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
        features = extract_mega_features(mega_model, sp_dataset, image_size=MEGA_IMAGE_SIZE)
        np.save(cache_file, features)
        print(f"  Cached to {cache_file}")

    mega_features_cache[species] = features
    print(f"  Shape: {features.shape}, Norm: {np.linalg.norm(features[0]):.3f}")

del mega_model
torch.cuda.empty_cache()
print("\\n\u2713 MegaDescriptor features extracted and cached")

# Apply query expansion to MegaDescriptor (same QE as MiewID)
mega_features_expanded = {}
for species, features in mega_features_cache.items():
    cfg = SPECIES_CONFIG[species]
    expanded = query_expansion(features, k=cfg["qe_k"])
    mega_features_expanded[species] = expanded
    print(f"{species}: MegaDesc QE with k={cfg['qe_k']}")

print("\u2713 Query expansion applied to MegaDescriptor features")
"""

# Insert MegaDescriptor cache cell AFTER query expansion cell (Cell 13)
# so that query_expansion() is already defined
insert_cell_after(
    nb,
    "Query expansion applied to global features",
    _MEGA_CACHE_CELL,
    "MegaDescriptor cache + QE",
)

# Insert MegaDescriptor model definition AFTER MiewID cache (Cell 12)
# so it appears between MiewID extraction and MegaDescriptor caching
insert_cell_after(
    nb,
    "Global features extracted and cached",
    _MEGA_MODEL_CELL,
    "MegaDescriptor model definition",
)


# =============================================================================
# Patch 4: compute_ensemble_similarity_matrix  (Cell 27, now shifted by +2)
# =============================================================================
# Replace function to use dual-global (miew_sim + mega_sim) instead of single global_sim.

_NEW_ENSEMBLE_FN = """\
# Cell 5.1: Ensemble Scoring Function

def compute_ensemble_similarity_matrix(species):
    \"\"\"Compute weighted ensemble similarity matrix (V4.0: dual-global).\"\"\"
    cfg = SPECIES_CONFIG[species]

    # Dual global similarity (SEPARATE matrices, not concatenated)
    miew_feats = global_features_expanded[species]
    miew_sim   = np.dot(miew_feats, miew_feats.T)

    mega_feats = mega_features_expanded[species]
    mega_sim   = np.dot(mega_feats, mega_feats.T)

    # Weighted dual-global
    ensemble_sim = cfg["miew_weight"] * miew_sim + cfg["mega_weight"] * mega_sim

    # Add weighted local match scores
    for extractor_name, weight in cfg["local_weights"].items():
        local_sim = match_scores_cache[species][extractor_name]
        ensemble_sim += weight * local_sim

    return ensemble_sim

print("\u2713 Ensemble scoring function defined (V4.0: dual-global)")
"""

replace_cell_source(
    nb,
    "# Cell 5.1: Ensemble Scoring Function",
    _NEW_ENSEMBLE_FN,
    "compute_ensemble_similarity_matrix dual-global",
)


# =============================================================================
# Patch 5: Ensemble cache print (Cell 28, shifted +2)
# =============================================================================
patch_cell(
    nb,
    "Using SIFT + KAZE + ALIKED (LightGlue) + MiewID v3 ensemble (V3.2)",
    "Using dual-global (MiewID v3 + MegaDescriptor-L) + SIFT/KAZE/ALIKED ensemble (V4.0)",
    "ensemble print update",
)


# =============================================================================
# Patch 6: Calibration cell  (Cell 30, shifted +2)
# =============================================================================
# This is the largest patch. We need to:
#   a) Keep AMI (ARI caused over-merging)
#   b) Add MIEW_W_GRID + MEGA_W_GRID
#   c) THR_GRID extended to 0.75 (31 steps)
#   d) KAZE explicit in grid
#   e) Add MegaDescriptor embedding extraction step
#   f) 5-loop grid search (mw, mgw, sw, kw, thr) with aw = residual
#   g) Update write-back for miew_weight/mega_weight
#   h) Update THL defaults

# --- 6a: Keep AMI import (ARI caused catastrophic Lynx over-merging) ---
# No change needed — AMI stays as-is from V3.2

# --- 6b: Calibration header comment ---
# Use shorter unique anchors to avoid Unicode dash length mismatch
patch_cell(
    nb,
    "# calibrate global_weight, sift_weight, kaze_weight and threshold_cluster\n"
    "# by grid-searching AMI against known training identities.",
    "# calibrate miew_weight, mega_weight, sift_weight, kaze_weight and\n"
    "# threshold_cluster by grid-searching AMI against known training identities.",
    "calibration header weights line",
)
patch_cell(
    nb,
    "#   1. Subsample \u2264500 training images per species.\n"
    "#   2. Extract MiewID global embeddings (GPU, TTA).\n"
    "#   3. Extract SIFT + KAZE descriptors (CPU, RootSIFT for SIFT).\n"
    "#   4. Compute BFMatcher match matrices using top-50 global preselection.\n"
    "#   4b. Extract ALIKED features (GPU) + LightGlue match matrix.\n"
    "#   5. Grid-search (gw, sw, aw, thr) with kw = 1-gw-sw-aw to maximise AMI.\n"
    "#   6. Write optimal weights + threshold back into SPECIES_CONFIG.",
    "#   1a. Extract MiewID global embeddings (GPU, TTA).\n"
    "#   1b. Extract MegaDescriptor global embeddings (GPU, TTA).\n"
    "#   2. Extract SIFT + KAZE descriptors (CPU, RootSIFT for SIFT).\n"
    "#   3. Compute BFMatcher match matrices using top-50 global preselection.\n"
    "#   3b. Extract ALIKED features (GPU) + LightGlue match matrix.\n"
    "#   4. Grid-search (mw, mgw, sw, kw, thr) with aw = residual to maximise AMI.\n"
    "#   5. Write optimal weights + threshold back into SPECIES_CONFIG.",
    "calibration pipeline steps",
)
patch_cell(
    nb,
    "# THL: no training split \u2192 weights + threshold unchanged from V2.5.",
    "# THL: no training split \u2192 weights + threshold unchanged from defaults.",
    "calibration THL note",
)

# --- 6c: Grid variables ---
patch_cell(
    nb,
    "GLOBAL_W_GRID   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]\n"
    "SIFT_W_GRID     = [0.0, 0.1, 0.2, 0.3]\n"
    "ALIKED_W_GRID   = [0.0, 0.1, 0.2, 0.3]\n"
    "THR_GRID        = [round(0.15 + i * 0.02, 2) for i in range(23)]  # 0.15..0.59",
    "MIEW_W_GRID     = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]\n"
    "MEGA_W_GRID     = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]\n"
    "SIFT_W_GRID     = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]\n"
    "KAZE_W_GRID     = [0, 0.05, 0.10, 0.15, 0.20]   # KAZE explicit (not residual)\n"
    "# aw = 1 - mw - mgw - sw - kw  (ALIKED is residual; AMI metric)\n"
    "THR_GRID        = np.linspace(0.15, 0.75, 31).round(4).tolist()  # 31 steps",
    "calibration grid variables",
)

# --- 6d: Calibration title print ---
patch_cell(
    nb,
    "V3.2 -- Joint Weight + Threshold Calibration (Training Identities)",
    "V4.0 -- Joint Weight + Threshold Calibration (Dual-Global + AMI)",
    "calibration title print",
)

# --- 6e: Grid summary print ---
patch_cell(
    nb,
    'print(f"  Grid: gw={GLOBAL_W_GRID}  sw={SIFT_W_GRID}  aw={ALIKED_W_GRID}")',
    'print(f"  Grid: mw={MIEW_W_GRID}  mgw={MEGA_W_GRID}  sw={SIFT_W_GRID}  kw={KAZE_W_GRID}")',
    "grid summary print",
)

# --- 6f: Model loading for calibration (add MegaDescriptor) ---
# Use just the unique code lines as anchor, avoid Unicode dashes
patch_cell(
    nb,
    "calib_model = get_global_model()\n"
    "calib_model.eval()\n"
    "\n"
    "calibrated_config = {}",
    "calib_model = get_global_model()\n"
    "calib_model.eval()\n"
    "\n"
    "calib_mega_model = get_mega_model()\n"
    "calib_mega_model.eval()\n"
    "\n"
    "calibrated_config = {}",
    "calibration model loading",
)

# --- 6g: Step 1 (MiewID) step numbering ---
patch_cell(
    nb,
    '  [1/5] Global embeddings: loaded from cache',
    '  [1/6] MiewID embeddings: loaded from cache',
    "step 1 cache label",
)
patch_cell(
    nb,
    '  [1/5] Extracting global embeddings...',
    '  [1/6] Extracting MiewID embeddings...',
    "step 1 extract label",
)

# --- 6h: Insert Step 1b (MegaDescriptor embeddings) after Step 1 ---
_STEP1B = (
    "    global_sim = np.clip(global_embs @ global_embs.T, 0.0, 1.0).astype(np.float32)\n"
    "\n"
    "    # Step 2: SIFT match matrix (cached)"
)
_STEP1B_NEW = (
    "    global_sim = np.clip(global_embs @ global_embs.T, 0.0, 1.0).astype(np.float32)\n"
    "\n"
    "    # Step 1b: MegaDescriptor global embeddings (cached)\n"
    "    _mgef = Path(f\"{_cp}_mega_global_embs.npy\")\n"
    "    if _mgef.exists():\n"
    "        print(f\"  [1b/6] MegaDescriptor embeddings: loaded from cache\")\n"
    "        mega_embs = np.load(_mgef)\n"
    "    else:\n"
    "        print(f\"  [1b/6] Extracting MegaDescriptor embeddings...\")\n"
    "        mega_embs = _extract_calib_embs(calib_mega_model, img_paths, 384)\n"
    "        np.save(_mgef, mega_embs)\n"
    "    mega_sim = np.clip(mega_embs @ mega_embs.T, 0.0, 1.0).astype(np.float32)\n"
    "\n"
    "    # Step 2: SIFT match matrix (cached)"
)
patch_cell(nb, _STEP1B, _STEP1B_NEW, "insert Step 1b MegaDescriptor calib embeddings")

# --- 6i: Step 2-4 renumbering ---
patch_cell(nb, "  [2/5] SIFT match matrix: loaded from cache", "  [2/6] SIFT match matrix: loaded from cache", "step 2 cache label")
patch_cell(nb, "  [2/5] Extracting SIFT descriptors + BFMatcher...", "  [2/6] Extracting SIFT descriptors + BFMatcher...", "step 2 extract label")
patch_cell(nb, "  [3/5] KAZE match matrix: loaded from cache", "  [3/6] KAZE match matrix: loaded from cache", "step 3 cache label")
patch_cell(nb, "  [3/5] Extracting KAZE descriptors + BFMatcher...", "  [3/6] Extracting KAZE descriptors + BFMatcher...", "step 3 extract label")
patch_cell(nb, "  [4/5] ALIKED matrix: loaded from cache", "  [4/6] ALIKED matrix: loaded from cache", "step 4 cache label")
patch_cell(nb, "  [4/5] Extracting ALIKED features + LightGlue matching...", "  [4/6] Extracting ALIKED features + LightGlue matching...", "step 4 extract label")

# --- 6j: Grid search section (Step 5 -> Step 5/6, dual-global + AMI) ---
# Replace from "# Step 5: grid search" through the end of the grid + calibrated_config assignment
_OLD_GRID = (
    "    # Step 5: grid search\n"
    "    from scipy.cluster.hierarchy import linkage as _sp_linkage, fcluster as _sp_fcluster\n"
    "    _n_calib = len(true_labels)\n"
    "    _triu_idx = np.triu_indices(_n_calib, k=1)\n"
    "\n"
    "    n_combos = sum(\n"
    "        1 for gw in GLOBAL_W_GRID for sw in SIFT_W_GRID\n"
    "        for aw in ALIKED_W_GRID if round(1.0 - gw - sw - aw, 4) >= 0\n"
    "    )\n"
    "    print(f\"  [5/5] Grid search ({n_combos} weight combos x {len(THR_GRID)} thresholds)...\")\n"
    "\n"
    "    best_gw  = cfg[\"global_weight\"]\n"
    "    best_sw  = cfg[\"local_weights\"][\"sift\"]\n"
    "    best_kw  = cfg[\"local_weights\"].get(\"kaze\", 0.0)\n"
    "    best_aw  = cfg[\"local_weights\"].get(\"aliked\", 0.0)\n"
    "    best_thr = cfg[\"threshold_cluster\"]\n"
    "    best_ami = -1.0\n"
    "\n"
    "    for gw in GLOBAL_W_GRID:\n"
    "        for sw in SIFT_W_GRID:\n"
    "            for aw in ALIKED_W_GRID:\n"
    "                kw = round(1.0 - gw - sw - aw, 4)\n"
    "                if kw < 0:\n"
    "                    continue\n"
    "                ensemble = gw * global_sim + sw * sift_matrix + kw * kaze_matrix + aw * aliked_matrix\n"
    "                dist_sq = np.clip(1.0 - ensemble, 0.0, 1.0).astype(np.float64)\n"
    "                # Compute linkage ONCE per weight combo; cut at all thresholds (~0s each)\n"
    "                Z = _sp_linkage(dist_sq[_triu_idx], method=\"average\")\n"
    "                for thr in THR_GRID:\n"
    "                    pred = _sp_fcluster(Z, t=thr, criterion=\"distance\")\n"
    "                    ami = _ami_score(true_labels, pred)\n"
    "                    if ami > best_ami:\n"
    "                        best_ami, best_gw, best_sw, best_kw, best_aw, best_thr = ami, gw, sw, kw, aw, thr\n"
    "\n"
    "    calibrated_config[sp] = {\n"
    "        \"global_weight\":  best_gw,\n"
    "        \"sift_weight\":    best_sw,\n"
    "        \"kaze_weight\":    best_kw,\n"
    "        \"aliked_weight\":  best_aw,\n"
    "        \"threshold\":      best_thr,\n"
    "        \"ami\":            best_ami,\n"
    "    }\n"
    "\n"
    "    prev_gw  = cfg[\"global_weight\"]\n"
    "    prev_sw  = cfg[\"local_weights\"][\"sift\"]\n"
    "    prev_kw  = cfg[\"local_weights\"].get(\"kaze\", 0.0)\n"
    "    prev_aw  = cfg[\"local_weights\"].get(\"aliked\", 0.0)\n"
    "    prev_thr = cfg[\"threshold_cluster\"]\n"
    "    print(f\"  weights:   ({prev_gw:.2f}, {prev_sw:.2f}, {prev_kw:.2f}, {prev_aw:.2f})\"\n"
    "          f\" -> ({best_gw:.2f}, {best_sw:.2f}, {best_kw:.2f}, {best_aw:.2f})\")\n"
    "    print(f\"  threshold: {prev_thr:.2f} -> {best_thr:.2f}\"\n"
    "          f\"  (AMI={best_ami:.4f}, t={_time.time()-t0:.1f}s)\")"
)

_NEW_GRID = (
    "    # Step 5: grid search (V4.0: dual-global + KAZE explicit + AMI)\n"
    "    from scipy.cluster.hierarchy import linkage as _sp_linkage, fcluster as _sp_fcluster\n"
    "    _n_calib = len(true_labels)\n"
    "    _triu_idx = np.triu_indices(_n_calib, k=1)\n"
    "\n"
    "    n_combos = sum(\n"
    "        1 for mw in MIEW_W_GRID for mgw in MEGA_W_GRID\n"
    "        for sw in SIFT_W_GRID for kw in KAZE_W_GRID\n"
    "        if round(1.0 - mw - mgw - sw - kw, 4) >= 0\n"
    "    )\n"
    "    print(f\"  [5/6] Grid search ({n_combos} weight combos x {len(THR_GRID)} thresholds)...\")\n"
    "\n"
    "    best_mw  = cfg[\"miew_weight\"]\n"
    "    best_mgw = cfg[\"mega_weight\"]\n"
    "    best_sw  = cfg[\"local_weights\"][\"sift\"]\n"
    "    best_kw  = cfg[\"local_weights\"].get(\"kaze\", 0.0)\n"
    "    best_aw  = cfg[\"local_weights\"].get(\"aliked\", 0.0)\n"
    "    best_thr = cfg[\"threshold_cluster\"]\n"
    "    best_ami = -1.0\n"
    "\n"
    "    for mw in MIEW_W_GRID:\n"
    "        for mgw in MEGA_W_GRID:\n"
    "            for sw in SIFT_W_GRID:\n"
    "                for kw in KAZE_W_GRID:\n"
    "                    aw = round(1.0 - mw - mgw - sw - kw, 4)\n"
    "                    if aw < 0:\n"
    "                        continue\n"
    "                    ensemble = (mw * global_sim + mgw * mega_sim\n"
    "                                + sw * sift_matrix + kw * kaze_matrix\n"
    "                                + aw * aliked_matrix)\n"
    "                    dist_sq = np.clip(1.0 - ensemble, 0.0, 1.0).astype(np.float64)\n"
    "                    Z = _sp_linkage(dist_sq[_triu_idx], method=\"average\")\n"
    "                    for thr in THR_GRID:\n"
    "                        pred = _sp_fcluster(Z, t=thr, criterion=\"distance\")\n"
    "                        ami = _ami_score(true_labels, pred)\n"
    "                        if ami > best_ami:\n"
    "                            best_ami = ami\n"
    "                            best_mw, best_mgw = mw, mgw\n"
    "                            best_sw, best_kw, best_aw = sw, kw, aw\n"
    "                            best_thr = thr\n"
    "\n"
    "    calibrated_config[sp] = {\n"
    "        \"miew_weight\":    best_mw,\n"
    "        \"mega_weight\":    best_mgw,\n"
    "        \"sift_weight\":    best_sw,\n"
    "        \"kaze_weight\":    best_kw,\n"
    "        \"aliked_weight\":  best_aw,\n"
    "        \"threshold\":      best_thr,\n"
    "        \"ami\":            best_ami,\n"
    "    }\n"
    "\n"
    "    prev_mw  = cfg[\"miew_weight\"]\n"
    "    prev_mgw = cfg[\"mega_weight\"]\n"
    "    prev_sw  = cfg[\"local_weights\"][\"sift\"]\n"
    "    prev_kw  = cfg[\"local_weights\"].get(\"kaze\", 0.0)\n"
    "    prev_aw  = cfg[\"local_weights\"].get(\"aliked\", 0.0)\n"
    "    prev_thr = cfg[\"threshold_cluster\"]\n"
    "    print(f\"  weights:   (mw={prev_mw:.2f}, mgw={prev_mgw:.2f}, sw={prev_sw:.2f}, kw={prev_kw:.2f}, aw={prev_aw:.2f})\"\n"
    "          f\" -> (mw={best_mw:.2f}, mgw={best_mgw:.2f}, sw={best_sw:.2f}, kw={best_kw:.2f}, aw={best_aw:.2f})\")\n"
    "    print(f\"  threshold: {prev_thr:.2f} -> {best_thr:.2f}\"\n"
    "          f\"  (AMI={best_ami:.4f}, t={_time.time()-t0:.1f}s)\")"
)

patch_cell(nb, _OLD_GRID, _NEW_GRID, "grid search dual-global + AMI")


# --- 6k: Delete calib_model (add mega) ---
patch_cell(
    nb,
    "del calib_model\n"
    "torch.cuda.empty_cache()",
    "del calib_model, calib_mega_model\n"
    "torch.cuda.empty_cache()",
    "delete calib models",
)


# --- 6l: THL defaults (dual-global keys) ---
patch_cell(
    nb,
    "# THL: no training split -- keep V2.5 values unchanged\n"
    "thl_cfg = SPECIES_CONFIG[\"TexasHornedLizards\"]\n"
    "calibrated_config[\"TexasHornedLizards\"] = {\n"
    "    \"global_weight\":  thl_cfg[\"global_weight\"],\n"
    "    \"sift_weight\":    thl_cfg[\"local_weights\"][\"sift\"],\n"
    "    \"kaze_weight\":    thl_cfg[\"local_weights\"][\"kaze\"],\n"
    "    \"aliked_weight\":  thl_cfg[\"local_weights\"][\"aliked\"],\n"
    "    \"threshold\":      thl_cfg[\"threshold_cluster\"],\n"
    "    \"ami\":            None,\n"
    "}",
    "# THL: no training split -- keep defaults unchanged\n"
    "thl_cfg = SPECIES_CONFIG[\"TexasHornedLizards\"]\n"
    "calibrated_config[\"TexasHornedLizards\"] = {\n"
    "    \"miew_weight\":    thl_cfg[\"miew_weight\"],\n"
    "    \"mega_weight\":    thl_cfg[\"mega_weight\"],\n"
    "    \"sift_weight\":    thl_cfg[\"local_weights\"][\"sift\"],\n"
    "    \"kaze_weight\":    thl_cfg[\"local_weights\"][\"kaze\"],\n"
    "    \"aliked_weight\":  thl_cfg[\"local_weights\"][\"aliked\"],\n"
    "    \"threshold\":      thl_cfg[\"threshold_cluster\"],\n"
    "    \"ami\":            None,\n"
    "}",
    "THL defaults dual-global",
)


# --- 6m: Write-back section ---
patch_cell(
    nb,
    "# Write calibrated weights + thresholds back into SPECIES_CONFIG (used by Cell 6.2)\n"
    "for sp, cal in calibrated_config.items():\n"
    "    SPECIES_CONFIG[sp][\"global_weight\"]         = cal[\"global_weight\"]\n"
    "    SPECIES_CONFIG[sp][\"local_weights\"][\"sift\"] = cal[\"sift_weight\"]\n"
    "    if \"kaze\" in SPECIES_CONFIG[sp][\"local_weights\"]:\n"
    "        SPECIES_CONFIG[sp][\"local_weights\"][\"kaze\"] = cal[\"kaze_weight\"]\n"
    "    if \"aliked\" in SPECIES_CONFIG[sp][\"local_weights\"]:\n"
    "        SPECIES_CONFIG[sp][\"local_weights\"][\"aliked\"] = cal[\"aliked_weight\"]\n"
    "    SPECIES_CONFIG[sp][\"threshold_cluster\"]     = cal[\"threshold\"]",
    "# Write calibrated weights + thresholds back into SPECIES_CONFIG (used by Cell 6.2)\n"
    "for sp, cal in calibrated_config.items():\n"
    "    SPECIES_CONFIG[sp][\"miew_weight\"]            = cal[\"miew_weight\"]\n"
    "    SPECIES_CONFIG[sp][\"mega_weight\"]            = cal[\"mega_weight\"]\n"
    "    SPECIES_CONFIG[sp][\"local_weights\"][\"sift\"]  = cal[\"sift_weight\"]\n"
    "    if \"kaze\" in SPECIES_CONFIG[sp][\"local_weights\"]:\n"
    "        SPECIES_CONFIG[sp][\"local_weights\"][\"kaze\"] = cal[\"kaze_weight\"]\n"
    "    if \"aliked\" in SPECIES_CONFIG[sp][\"local_weights\"]:\n"
    "        SPECIES_CONFIG[sp][\"local_weights\"][\"aliked\"] = cal[\"aliked_weight\"]\n"
    "    SPECIES_CONFIG[sp][\"threshold_cluster\"]      = cal[\"threshold\"]",
    "write-back dual-global",
)


# --- 6n: Calibrated config print (dual-global format) ---
patch_cell(
    nb,
    "print(\"\\nCalibrated SPECIES_CONFIG:\")\n"
    "for sp, cfg in SPECIES_CONFIG.items():\n"
    "    lw = cfg[\"local_weights\"]\n"
    "    print(f\"  {sp:<25}: gw={cfg['global_weight']:.2f}\"\n"
    "          f\"  sw={lw.get('sift', 0):.2f}\"\n"
    "          f\"  kw={lw.get('kaze', 0):.2f}\"\n"
    "          f\"  aw={lw.get('aliked', 0):.2f}\"\n"
    "          f\"  thr={cfg['threshold_cluster']:.2f}\")",
    "print(\"\\nCalibrated SPECIES_CONFIG:\")\n"
    "for sp, cfg in SPECIES_CONFIG.items():\n"
    "    lw = cfg[\"local_weights\"]\n"
    "    print(f\"  {sp:<25}: mw={cfg['miew_weight']:.2f}\"\n"
    "          f\"  mgw={cfg['mega_weight']:.2f}\"\n"
    "          f\"  sw={lw.get('sift', 0):.2f}\"\n"
    "          f\"  kw={lw.get('kaze', 0):.2f}\"\n"
    "          f\"  aw={lw.get('aliked', 0):.2f}\"\n"
    "          f\"  thr={cfg['threshold_cluster']:.2f}\")",
    "calibrated config print dual-global",
)


# =============================================================================
# Patch 7: Weighted Voting Summary (Cell 29, shifted +2)
# =============================================================================
# This cell uses cfg['global_weight'] which no longer exists. Replace whole cell.
_NEW_VOTING_SUMMARY = """\
# Cell 5.3: Weighted Voting Summary

print("Ensemble Weights Summary (V4.0: dual-global):\\n")
for species, cfg in SPECIES_CONFIG.items():
    print(f"{species}:")
    print(f"  MiewID v3: {cfg['miew_weight']:.0%}")
    print(f"  MegaDescriptor-L: {cfg['mega_weight']:.0%}")
    for extractor, weight in cfg['local_weights'].items():
        print(f"  {extractor.upper()}: {weight:.0%}")
    print()

print("\u2713 Ensemble voting configured")
"""
replace_cell_source(
    nb,
    "# Cell 5.3: Weighted Voting Summary",
    _NEW_VOTING_SUMMARY,
    "voting summary dual-global",
)


# =============================================================================
# Patch 8a: Fix ROOT_DIR path (Kaggle changed competition mount point)
# =============================================================================
patch_cell(
    nb,
    'ROOT_DIR = "/kaggle/input/animal-clef-2026"',
    'ROOT_DIR = "/kaggle/input/animal-clef-2026"\n'
    'if not os.path.exists(ROOT_DIR):\n'
    '    ROOT_DIR = "/kaggle/input/competitions/animal-clef-2026"\n'
    'assert os.path.exists(ROOT_DIR), f"Competition data not found at {ROOT_DIR}"',
    "ROOT_DIR fallback path",
)


# =============================================================================
# Patch 8: Add timm import in imports cell (Cell 3) for MegaDescriptor
# =============================================================================
# timm is already imported if it was used before. Let's check and add if needed.
# Actually, timm is used in MiewID loading via AutoModel, but MegaDescriptor
# uses timm.create_model directly. We need to ensure `import timm` exists.
patch_cell(
    nb,
    "import torch.nn as nn",
    "import torch.nn as nn\nimport timm",
    "add timm import",
)


# =============================================================================
# Patch 9: Add numpy import for linspace in calibration
# =============================================================================
# np is already available from the imports cell, but THR_GRID uses np.linspace
# which is fine since numpy is imported as np in Cell 3.
# No patch needed.


# =============================================================================
# Patch 10: V3 cache block — skip mega_global_embs (new, must be fresh)
# =============================================================================
# The V3 calib_cache won't have _mega_global_embs.npy files, but the copy loop
# will harmlessly copy whatever exists (no _mega files = nothing extra copied).
# Embeddings cache: V3 won't have *_mega.npy files either. Safe as-is.
# No patch needed — the existing copy logic handles this correctly.


# =============================================================================
# Patch 11: Section 3 markdown header cleanup
# =============================================================================
# The old header mentions "SuperPoint" and "DISK" from the original template.
patch_cell(
    nb,
    "## Section 3: Local Features (SIFT, SuperPoint, ALIKED, DISK)",
    "## Section 3: Local Features (SIFT, KAZE, ALIKED)",
    "section 3 header",
)


# =============================================================================
# Patch 12: Markdown header in Cell 0
# =============================================================================
patch_cell(
    nb,
    "Combines global features (MiewID v3) with species-specific local features (SIFT, SuperPoint, ALIKED, DISK)",
    "Combines dual global features (MiewID v3 + MegaDescriptor-L-384) with species-specific local features (SIFT, KAZE, ALIKED)",
    "Cell 0 approach description",
)


# =============================================================================
# Write output
# =============================================================================
DST.parent.mkdir(exist_ok=True)

with open(DST, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {DST}  ({len(nb['cells'])} cells)")


# =============================================================================
# Verification
# =============================================================================
print("\nVerification:")

with open(DST) as f:
    out_text = f.read()

nb_out = json.loads(out_text)
all_src = ""
for cell in nb_out["cells"]:
    s = cell["source"]
    all_src += (s if isinstance(s, str) else "".join(s)) + "\n"

checks = [
    # ── Patch 0+1: version comment (in SPECIES_CONFIG cell) ──
    ("version comment V4.0",
     "V4.0: Dual-global (MiewID v3 + MegaDescriptor-L-384) + AMI calibration"),
    ("OLD V3.2 comment gone",
     "V3.2: Fix calib/test score mismatch"),

    # ── Patch 1: SPECIES_CONFIG ──
    ("miew_weight in config",
     '"miew_weight":'),
    ("mega_weight in config",
     '"mega_weight":'),
    ("OLD global_weight gone",
     '"global_weight":'),
    ("weight sum assertion updated",
     'cfg["miew_weight"] + cfg["mega_weight"] + sum(cfg["local_weights"].values())'),
    ("Lynx threshold 0.65",
     '"threshold_cluster": 0.65'),
    ("Salamander sift 0.30",
     '"sift": 0.30, "kaze": 0.15, "aliked": 0.05'),
    ("THL aliked 0.20",
     '"sift": 0.15, "kaze": 0.10, "aliked": 0.20'),

    # ── Patch 3: MegaDescriptor cells ──
    ("MegaDescriptor model function",
     "def get_mega_model():"),
    ("MegaDescriptor extraction function",
     "def extract_mega_features(model, dataset, image_size=384):"),
    ("MegaDescriptor num_classes=0",
     "num_classes=0"),
    ("MegaDescriptor cache file",
     'cache/embeddings/{species}_mega.npy'),
    ("mega_features_expanded dict",
     "mega_features_expanded = {}"),
    ("MegaDescriptor QE",
     "MegaDesc QE with k="),
    ("timm create_model for MegaDescriptor",
     "hf-hub:BVRA/MegaDescriptor-L-384"),
    ("MEGA_IMAGE_SIZE defined",
     "MEGA_IMAGE_SIZE = 384"),

    # ── Patch 4: compute_ensemble_similarity_matrix ──
    ("ensemble uses miew_weight",
     'cfg["miew_weight"] * miew_sim'),
    ("ensemble uses mega_weight",
     'cfg["mega_weight"] * mega_sim'),
    ("ensemble uses mega_features_expanded",
     "mega_features_expanded[species]"),
    ("OLD single global_sim gone",
     'cfg["global_weight"] * global_sim'),

    # ── Patch 6: Calibration (AMI kept, ARI reverted) ──
    ("AMI import preserved",
     "adjusted_mutual_info_score as _ami_score"),
    ("OLD ARI import gone",
     "adjusted_rand_score as _ari_score"),
    ("MIEW_W_GRID defined",
     "MIEW_W_GRID"),
    ("MEGA_W_GRID defined",
     "MEGA_W_GRID"),
    ("KAZE_W_GRID defined (explicit)",
     "KAZE_W_GRID"),
    ("THR_GRID extended to 0.75",
     "np.linspace(0.15, 0.75, 31)"),
    ("OLD THR_GRID gone",
     "for i in range(23)"),
    ("calib MegaDescriptor model loaded",
     "calib_mega_model = get_mega_model()"),
    ("calib MegaDescriptor embeddings cached",
     "_mega_global_embs.npy"),
    ("4-loop grid search (mw)",
     "for mw in MIEW_W_GRID:"),
    ("4-loop grid search (mgw)",
     "for mgw in MEGA_W_GRID:"),
    ("4-loop grid search (kw explicit)",
     "for kw in KAZE_W_GRID:"),
    ("aw is residual",
     "aw = round(1.0 - mw - mgw - sw - kw, 4)"),
    ("AMI in grid search",
     "ami = _ami_score(true_labels, pred)"),
    ("calibrated_config has miew_weight",
     '"miew_weight":    best_mw'),
    ("calibrated_config has mega_weight",
     '"mega_weight":    best_mgw'),
    ("THL defaults has miew_weight",
     'thl_cfg["miew_weight"]'),
    ("THL defaults has mega_weight",
     'thl_cfg["mega_weight"]'),
    ("write-back miew_weight",
     'SPECIES_CONFIG[sp]["miew_weight"]'),
    ("write-back mega_weight",
     'SPECIES_CONFIG[sp]["mega_weight"]'),
    ("calib models deleted together",
     "del calib_model, calib_mega_model"),
    ("grid print shows mw/mgw",
     "mw={MIEW_W_GRID}  mgw={MEGA_W_GRID}"),
    ("step 1b label",
     "[1b/6] MegaDescriptor embeddings"),

    # ── V3.2 structure preserved ──
    ("lightglue install preserved",
     "git+https://github.com/cvg/LightGlue.git"),
    ("ALIKEDExtractor preserved",
     "from lightglue import ALIKED as _ALIKED_Extractor"),
    ("LightGlue scoring preserved",
     "def compute_local_score_lightglue(feat1, feat2):"),
    ("ratio_only function preserved",
     "def compute_local_score_ratio_only(feat1, feat2"),
    ("ALIKED seg_mask filter preserved",
     "ALIKED extractor defined (LightGlue, seg_mask filtered)"),
    ("V2.8 yellow mask preserved",
     "if 'SalamanderID2025' in rel_key:"),
    ("KAZEExtractor preserved",
     "class KAZEExtractor"),
    ("SIFTExtractor preserved",
     "class SIFTExtractor"),
    ("ensemble cache refresh preserved",
     "ensemble_similarity_cache refreshed"),
    ("timm import added",
     "import timm"),

    # ── Patch 11/12: Section headers ──
    ("section 3 header updated",
     "## Section 3: Local Features (SIFT, KAZE, ALIKED)"),
    ("section 2 header updated",
     "## Section 2: Global Features (MiewID v3 + MegaDescriptor-L-384)"),
]

ABSENT = {
    "OLD V3.2 comment gone",
    "OLD global_weight gone",
    "OLD single global_sim gone",
    "OLD ARI import gone",
    "OLD THR_GRID gone",
}

passed = 0
failed = 0
for name, pattern in checks:
    present     = pattern in all_src
    must_absent = name in ABSENT
    ok = (not present) if must_absent else present
    mark = "\u2713" if ok else "\u2717"
    if not ok:
        failed += 1
        print(f"  {mark} FAIL [{name}]  "
              f"{'(should be absent)' if must_absent else '(missing)'}: "
              f"{repr(pattern[:70])}")
    else:
        passed += 1
        print(f"  {mark}  {name}")

print(f"\n{passed}/{passed+failed} checks passed")
if failed:
    sys.exit(1)
else:
    print("All checks passed \u2014 V4.0 notebook ready for submission.")
