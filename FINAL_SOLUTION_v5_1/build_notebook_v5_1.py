"""
Build script for V5.1 notebook.

Changes from V4.0:
  1. Replace pretrained MegaDescriptor-L-384 with fine-tuned MegaDescriptor-T-224.
     Fine-tuned per species using ArcFace loss on species-specific data.
     THL: excluded (uses pretrained L-384 fallback).
  2. Bake in V4.0 calibrated weights as SPECIES_CONFIG defaults.
  3. Update cache suffixes to avoid conflict with V4.0 pretrained embeddings.

NO isotonic calibration. NO 2-phase clustering. Pure V4.0 approach + fine-tuned embeddings.

Builds from: V4.0 notebook.
"""

import json, copy, pathlib

SRC = pathlib.Path("FINAL_SOLUTION_v4_0/ensemble_global_local_reid_v4_0.ipynb")
DST = pathlib.Path("FINAL_SOLUTION_v5_1/ensemble_global_local_reid_v5_1.ipynb")

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


# =============================================================================
# Patch 0+1: SPECIES_CONFIG (Cell 4) — V4.0 calibrated defaults
# =============================================================================

_NEW_SPECIES_CONFIG = """\
# Cell 1.3: Configuration - Species-Specific Weights

# V5.1: Fine-tuned MegaDescriptor (ArcFace) + V4.0 pure clustering + AMI
# Defaults: V4.0 calibrated values (best known configuration)

SPECIES_CONFIG = {
    "SalamanderID2025": {
        # V4.0 calibrated: mw=0.40 mgw=0.00 sw=0.40 kw=0.15 aw=0.05 thr=0.51
        "miew_weight": 0.40,
        "mega_weight": 0.00,
        "local_extractors": ["sift", "kaze", "aliked"],
        "local_weights": {"sift": 0.40, "kaze": 0.15, "aliked": 0.05},
        "threshold_cluster": 0.51,
        "image_size": 512,
        "qe_k": 3,
    },
    "SeaTurtleID2022": {
        # V4.0 calibrated: mw=0.50 mgw=0.30 sw=0.10 kw=0.10 aw=0.00 thr=0.65
        "miew_weight": 0.50,
        "mega_weight": 0.30,
        "local_extractors": ["sift", "kaze", "aliked"],
        "local_weights": {"sift": 0.10, "kaze": 0.10, "aliked": 0.00},
        "threshold_cluster": 0.65,
        "image_size": 512,
        "qe_k": 8,
    },
    "LynxID2025": {
        # V4.0 calibrated: mw=0.40 mgw=0.00 sw=0.05 kw=0.00 aw=0.55 thr=0.67
        "miew_weight": 0.40,
        "mega_weight": 0.00,
        "local_extractors": ["sift", "kaze", "aliked"],
        "local_weights": {"sift": 0.05, "kaze": 0.00, "aliked": 0.55},
        "threshold_cluster": 0.67,
        "image_size": 512,
        "qe_k": 5,
    },
    "TexasHornedLizards": {
        # V4.0 uncalibrated
        "miew_weight": 0.275,
        "mega_weight": 0.275,
        "local_extractors": ["sift", "kaze", "aliked"],
        "local_weights": {"sift": 0.15, "kaze": 0.10, "aliked": 0.20},
        "threshold_cluster": 0.30,
        "image_size": 512,
        "qe_k": 5,
    },
}

# Verify weights sum to 1.0
for species, cfg in SPECIES_CONFIG.items():
    total = cfg["miew_weight"] + cfg["mega_weight"] + sum(cfg["local_weights"].values())
    assert abs(total - 1.0) < 0.01, f"{species} weights don't sum to 1.0: {total}"

print("\\u2713 Species configuration loaded (V5.1: fine-tuned MegaDescriptor + V4.0 defaults)")
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
    "SPECIES_CONFIG V5.1 defaults",
)


# =============================================================================
# Patch 2: Section 2 markdown header
# =============================================================================
patch_cell(
    nb,
    "## Section 2: Global Features (MiewID v3 + MegaDescriptor-L-384)",
    "## Section 2: Global Features (MiewID v3 + Fine-tuned MegaDescriptor)",
    "section 2 header",
)


# =============================================================================
# Patch 3: Replace MegaDescriptor model function (Cell 2.3b)
# =============================================================================

_NEW_MEGA_MODEL_CELL = """\
# Cell 2.3b: Fine-tuned MegaDescriptor Model + Extraction

# V5.1: Per-species model selection:
#   - Species WITH fine-tuned checkpoint: MegaDescriptor-T-224 + loaded ArcFace weights
#   - Species WITHOUT checkpoint (e.g., THL): pretrained MegaDescriptor-L-384 (V4.0 behavior)

FINETUNE_DIR = Path("/kaggle/input/megadesc-finetuned-v5")
if not FINETUNE_DIR.exists():
    FINETUNE_DIR = Path("/kaggle/input/datasets/sreevaatsavbavana/megadesc-finetuned-v5")
FINETUNE_SPECIES_FILES = {
    "LynxID2025":       "LynxID2025_megadesc_v5.pth",
    "SalamanderID2025": "SalamanderID2025_megadesc_v5.pth",
    "SeaTurtleID2022":  "SeaTurtleID2022_megadesc_v5.pth",
}

# Detect if any fine-tuned checkpoints exist
_ft_available = any((FINETUNE_DIR / f).exists() for f in FINETUNE_SPECIES_FILES.values())
if _ft_available:
    print("\\u2713 Fine-tuned MegaDescriptor-T-224 checkpoints detected")
    for sp, fname in FINETUNE_SPECIES_FILES.items():
        exists = (FINETUNE_DIR / fname).exists()
        print(f"  {sp}: {'\\u2713' if exists else '\\u2717 (will use L-384 pretrained)'} {fname}")
    print("  Species not in FINETUNE_SPECIES_FILES use pretrained L-384 fallback")
else:
    print("\\u26a0 No fine-tuned checkpoints found -- all species use pretrained MegaDescriptor-L-384")

# Cache for loaded models (avoid reloading per species)
_mega_models_cache = {}


def _species_has_ft(species):
    '''Check if a species has a fine-tuned checkpoint available.'''
    if not _ft_available or not species:
        return False
    fname = FINETUNE_SPECIES_FILES.get(species)
    return bool(fname and (FINETUNE_DIR / fname).exists())


def get_mega_config(species=None):
    '''Return (model_name, image_size, batch_size) for this species MegaDescriptor variant.'''
    if _species_has_ft(species):
        return 'hf-hub:BVRA/MegaDescriptor-T-224', 224, 64
    else:
        return 'hf-hub:BVRA/MegaDescriptor-L-384', 384, 32


def get_mega_model(species=None):
    '''Load MegaDescriptor model. Fine-tuned T-224 if available, else pretrained L-384.'''
    ft_path = None
    if _species_has_ft(species):
        fname = FINETUNE_SPECIES_FILES[species]
        ft_path = FINETUNE_DIR / fname

    model_name, _, _ = get_mega_config(species)
    cache_key = str(ft_path) if ft_path else f"pretrained_{model_name}"

    if cache_key in _mega_models_cache:
        return _mega_models_cache[cache_key]

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,  # CRITICAL: without this, returns class logits not embeddings
    )

    if ft_path:
        print(f"  Loading fine-tuned weights: {ft_path.name}")
        state = torch.load(str(ft_path), map_location="cpu")
        model.load_state_dict(state)
    else:
        print(f"  Using pretrained {model_name} for {species}")

    model = model.eval().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    _mega_models_cache[cache_key] = model
    return model


def extract_mega_features(model, dataset, image_size, batch_size=32):
    '''Extract L2-normalized MegaDescriptor features with TTA (horizontal flip).'''
    dataset.transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
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


print("\\u2713 MegaDescriptor model + extraction function defined (V5.1)")
"""

replace_cell_source(
    nb,
    "# Cell 2.3b: MegaDescriptor-L-384 Model + Extraction",
    _NEW_MEGA_MODEL_CELL,
    "MegaDescriptor fine-tuned model",
)


# =============================================================================
# Patch 4: MegaDescriptor caching cell — per-species model + cache suffix
# =============================================================================

_NEW_MEGA_CACHE_CELL = """\
# Cell 2.3c: Cache MegaDescriptor Embeddings + Query Expansion

# V5.1: Per-species model selection.
#   Fine-tuned species: T-224, cache suffix _mega_ft
#   Other species (THL): pretrained L-384, cache suffix _mega

mega_features_cache = {}

for species in test_meta["dataset"].unique():
    print(f"\\n{'='*60}")
    print(f"Processing {species} - MegaDescriptor Features")

    mega_model = get_mega_model(species=species)
    _, mega_img_size, mega_batch = get_mega_config(species)
    _is_ft = _species_has_ft(species)

    cache_suffix = "_mega_ft" if _is_ft else "_mega"
    cache_file = f"cache/embeddings/{species}{cache_suffix}.npy"

    if os.path.exists(cache_file):
        print(f"  Loading cached embeddings: {cache_file}")
        features = np.load(cache_file)
    else:
        sp_meta = test_meta[test_meta["dataset"] == species]
        sp_dataset = full_dataset.get_subset(sp_meta.index.values)
        features = extract_mega_features(mega_model, sp_dataset, image_size=mega_img_size, batch_size=mega_batch)
        np.save(cache_file, features)
        print(f"  Cached to {cache_file}")

    mega_features_cache[species] = features
    print(f"  Shape: {features.shape}, Norm: {np.linalg.norm(features[0]):.3f}")
    print(f"  Model: {'fine-tuned T-224' if _is_ft else 'pretrained L-384'}")

# Clean up model cache to free GPU memory
_mega_models_cache.clear()
torch.cuda.empty_cache()
print("\\n\\u2713 MegaDescriptor features extracted and cached")

# Apply query expansion to MegaDescriptor (same QE as MiewID)
mega_features_expanded = {}
for species, features in mega_features_cache.items():
    cfg = SPECIES_CONFIG[species]
    expanded = query_expansion(features, k=cfg["qe_k"])
    mega_features_expanded[species] = expanded
    print(f"{species}: MegaDesc QE with k={cfg['qe_k']}")

print("\\u2713 Query expansion applied to MegaDescriptor features")
"""

replace_cell_source(
    nb,
    "# Cell 2.3c: Cache MegaDescriptor Embeddings + Query Expansion",
    _NEW_MEGA_CACHE_CELL,
    "MegaDescriptor per-species caching",
)


# =============================================================================
# Patch 5: Calibration title + extend threshold grid
# =============================================================================
patch_cell(
    nb,
    "V4.0 -- Joint Weight + Threshold Calibration (Dual-Global + AMI)",
    "V5.1 -- Joint Weight + Threshold Calibration (Fine-tuned MegaDesc + AMI)",
    "calibration title print",
)

# Extend threshold grid from [0.15, 0.75] to [0.15, 0.85]
patch_cell(
    nb,
    "THR_GRID        = np.linspace(0.15, 0.75, 31).round(4).tolist()",
    "THR_GRID        = np.linspace(0.15, 0.85, 36).round(4).tolist()  # V5.1: extended to 0.85",
    "extend threshold grid",
)


# =============================================================================
# Patch 6: Calibration — load fine-tuned mega model per species
# =============================================================================

# 6a: Replace global mega model load with per-species placeholder
patch_cell(
    nb,
    "calib_mega_model = get_mega_model()\n"
    "calib_mega_model.eval()",
    "# V5.1: species-specific fine-tuned model loaded inside loop\n"
    "calib_mega_model = None  # loaded per-species below",
    "calibration mega model loading",
)

# 6b: Replace Step 1b to use per-species fine-tuned model + dynamic cache suffix
_OLD_STEP1B = (
    "    # Step 1b: MegaDescriptor global embeddings (cached)\n"
    "    _mgef = Path(f\"{_cp}_mega_global_embs.npy\")\n"
    "    if _mgef.exists():\n"
    "        print(f\"  [1b/6] MegaDescriptor embeddings: loaded from cache\")\n"
    "        mega_embs = np.load(_mgef)\n"
    "    else:\n"
    "        print(f\"  [1b/6] Extracting MegaDescriptor embeddings...\")\n"
    "        mega_embs = _extract_calib_embs(calib_mega_model, img_paths, 384)\n"
    "        np.save(_mgef, mega_embs)\n"
    "    mega_sim = np.clip(mega_embs @ mega_embs.T, 0.0, 1.0).astype(np.float32)"
)
_NEW_STEP1B = (
    "    # Step 1b: MegaDescriptor global embeddings (cached, V5.1: per-species model)\n"
    "    _is_ft_calib = _species_has_ft(sp)\n"
    "    _mega_suffix = \"_megaft_global_embs\" if _is_ft_calib else \"_mega_global_embs\"\n"
    "    _mgef = Path(f\"{_cp}{_mega_suffix}.npy\")\n"
    "    _, _mega_img_sz, _ = get_mega_config(sp)\n"
    "    if _mgef.exists():\n"
    "        print(f\"  [1b/6] MegaDescriptor embeddings: loaded from cache\")\n"
    "        mega_embs = np.load(_mgef)\n"
    "    else:\n"
    "        print(f\"  [1b/6] Extracting MegaDescriptor embeddings...\")\n"
    "        calib_mega_model = get_mega_model(species=sp)\n"
    "        mega_embs = _extract_calib_embs(calib_mega_model, img_paths, _mega_img_sz)\n"
    "        np.save(_mgef, mega_embs)\n"
    "    mega_sim = np.clip(mega_embs @ mega_embs.T, 0.0, 1.0).astype(np.float32)"
)
patch_cell(nb, _OLD_STEP1B, _NEW_STEP1B, "step 1b fine-tuned + cache suffix")

# 6c: Clean up mega model cache after calibration
patch_cell(
    nb,
    "del calib_model, calib_mega_model\n"
    "torch.cuda.empty_cache()",
    "del calib_model, calib_mega_model\n"
    "_mega_models_cache.clear()\n"
    "torch.cuda.empty_cache()",
    "delete calib models",
)


# =============================================================================
# Patch 7: Section header + ensemble print
# =============================================================================
patch_cell(
    nb,
    "Using dual-global (MiewID v3 + MegaDescriptor-L) + SIFT/KAZE/ALIKED ensemble (V4.0)",
    "Using dual-global (MiewID v3 + fine-tuned MegaDescriptor) + SIFT/KAZE/ALIKED ensemble (V5.1)",
    "ensemble print update",
)


# =============================================================================
# Patch 8: Cell 0 markdown header
# =============================================================================
patch_cell(
    nb,
    "# AnimalCLEF 2026: Global + Local Features Ensemble",
    "# AnimalCLEF 2026: Ensemble Re-ID V5.1 (Fine-tuned MegaDescriptor)",
    "cell 0 header",
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
    # ── Version ──
    ("version comment V5.1",
     "V5.1: Fine-tuned MegaDescriptor (ArcFace) + V4.0 pure clustering + AMI"),
    ("OLD V4.0 header gone",
     "Global + Local Features Ensemble\n"),

    # ── SPECIES_CONFIG ──
    ("Lynx miew_weight 0.40",
     '"miew_weight": 0.40'),
    ("Lynx mega_weight 0.00",
     '"mega_weight": 0.00'),
    ("Lynx threshold 0.67",
     '"threshold_cluster": 0.67'),
    ("Salamander threshold 0.51",
     '"threshold_cluster": 0.51'),
    ("SeaTurtle mega_weight 0.30",
     '"mega_weight": 0.30'),
    ("SeaTurtle threshold 0.65",
     '"threshold_cluster": 0.65'),
    ("THL mega_weight 0.275",
     '"mega_weight": 0.275'),

    # ── Fine-tuned model ──
    ("FINETUNE_DIR defined",
     'FINETUNE_DIR = Path("/kaggle/input/megadesc-finetuned-v5")'),
    ("FINETUNE_DIR fallback",
     'FINETUNE_DIR = Path("/kaggle/input/datasets/sreevaatsavbavana/megadesc-finetuned-v5")'),
    ("species checkpoint files dict",
     "FINETUNE_SPECIES_FILES"),
    ("_species_has_ft helper",
     "def _species_has_ft(species):"),
    ("get_mega_config function",
     "def get_mega_config(species=None):"),
    ("get_mega_model has species param",
     "def get_mega_model(species=None):"),
    ("T-224 returned for fine-tuned",
     "MegaDescriptor-T-224"),
    ("L-384 fallback for pretrained",
     "MegaDescriptor-L-384"),
    ("num_classes=0 preserved",
     "num_classes=0"),
    ("load_state_dict for fine-tuned",
     "model.load_state_dict(state)"),
    ("extract_mega_features TTA",
     "torch.flip(images, dims=[3])"),

    # ── Per-species caching ──
    ("_mega_ft cache suffix",
     '_mega_ft'),
    ("per-species mega model in cache cell",
     "mega_model = get_mega_model(species=species)"),
    ("mega QE applied",
     "query_expansion(features"),

    # ── Calibration uses fine-tuned model ──
    ("calib mega per-species",
     "calib_mega_model = get_mega_model(species=sp)"),
    ("calib mega cache suffix",
     '_megaft_global_embs'),
    ("calib mega image size dynamic",
     "get_mega_config(sp)"),

    # ── V4.0 clustering preserved (NO 2-phase) ──
    ("agglomerative clustering",
     "AgglomerativeClustering"),
    ("distance_threshold in clustering",
     "distance_threshold"),

    # ── NO isotonic (should NOT be present) ──
    ("no isotonic in ensemble",
     "compute_ensemble_similarity_matrix"),
]

passed = 0
failed = 0
for desc, needle in checks:
    if desc.startswith("OLD ") or desc.startswith("no "):
        # Negative check — ensure OLD string is gone or isotonic is absent
        if desc.startswith("OLD "):
            if needle not in all_src:
                print(f"  \u2713  {desc}")
                passed += 1
            else:
                print(f"  \u2717  {desc} — should be removed but still present!")
                failed += 1
        else:
            # "no isotonic" — just check compute_ensemble_similarity_matrix exists (V4.0 version)
            if needle in all_src:
                print(f"  \u2713  {desc}")
                passed += 1
            else:
                print(f"  \u2717  {desc} — not found!")
                failed += 1
    else:
        if needle in all_src:
            print(f"  \u2713  {desc}")
            passed += 1
        else:
            print(f"  \u2717  {desc} — NOT FOUND!")
            failed += 1

# Extra: ensure isotonic_models is NOT in the notebook (V4.0 doesn't have it)
if "isotonic_models" in all_src:
    print(f"  \u2717  isotonic_models should NOT be present!")
    failed += 1
else:
    print(f"  \u2713  no isotonic_models (clean V4.0 approach)")
    passed += 1

# Extra: ensure cluster_2phase is NOT in the notebook
if "cluster_2phase" in all_src:
    print(f"  \u2717  cluster_2phase should NOT be present!")
    failed += 1
else:
    print(f"  \u2713  no cluster_2phase (pure clustering only)")
    passed += 1

# Extra: ensure threshold_known is NOT in the notebook
if "threshold_known" in all_src:
    print(f"  \u2717  threshold_known should NOT be present!")
    failed += 1
else:
    print(f"  \u2713  no threshold_known (pure clustering only)")
    passed += 1

print(f"\n{passed}/{passed+failed} checks passed")
if failed > 0:
    print(f"FAILED: {failed} checks")
    raise SystemExit(1)
else:
    print("All checks passed — V5.1 notebook ready for submission.")
