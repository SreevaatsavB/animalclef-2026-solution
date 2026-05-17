"""
Build script for V3.3 notebook.

Changes from V3.2 (and V3.2.2 live fixes baked in):
  1. SPECIES_CONFIG: replace V3.1-era defaults with V3.2.2 calibrated values.
       Lynx:  gw=0.40 sw=0.00 kw=0.05 aw=0.55 thr=0.65
       Sal:   gw=0.50 sw=0.30 kw=0.15 aw=0.05 thr=0.47
       Turtle:gw=0.70 sw=0.20 kw=0.05 aw=0.05 thr=0.57
       THL:   gw=0.55 sw=0.15 kw=0.10 aw=0.20 thr=0.30
       All sums to 1.00 with spw=0.00 (no SuperPoint contribution initially).
  2. Calibration grid fix (V3.2.2 structure):
       - KAZE_W_GRID added as EXPLICIT dimension (was residual in V3.2).
       - SIFT_W_GRID ceiling raised to 0.40, KAZE_W_GRID ceiling to 0.20.
       - ALIKED_W_GRID extended to cover 0.55/0.60 (Lynx aw=0.55 was uncalibrated).
       - THR_GRID extended to 0.75 (np.linspace, 31 steps; was capped at 0.59).
       - SuperPoint (spw) is the residual: spw = 1 - gw - sw - kw - aw.
  3. Add SuperPointExtractor class (Patch A).
       - LightGlue SuperPoint, max_num_keypoints=2048, 256-dim descriptors.
       - apply get_seg_mask() filter (same as ALIKEDExtractor).
       - stores image_size from img.shape[-2:] for LightGlue.
  4. Add _get_sp_matcher() + compute_local_score_sp_lg() (Patch B).
       - Mirrors _get_aliked_matcher() / compute_local_score_lightglue() exactly.
  5. Add 'superpoint' to get_extractor() factory (Patch C).
  6. Update compute_pairwise_matches_fast():
       - cache suffix: superpoint → _sp_lg_matches (Patch D cache suffix).
       - dispatch: superpoint uses compute_local_score_sp_lg() (Patch D dispatch).
  7. Add V3.3 cache preload block (SP features + SP calib matrices).
  8. Calibration: add SuperPoint extraction + LightGlue matrix as Step 5.
  9. Calibration grid: 4 explicit loops (gw, sw, kw, aw); spw = residual.
 10. Calibration: update write-back to include 'superpoint' key.
 11. Cell 16 print updated for V3.3.
"""

import json, copy, pathlib, sys

SRC = pathlib.Path("FINAL_SOLUTION_v3_2/ensemble_global_local_reid_v3_2.ipynb")
DST = pathlib.Path("FINAL_SOLUTION_v3_3/ensemble_global_local_reid_v3_3.ipynb")

assert SRC.exists(), f"Source notebook not found: {SRC}"

with open(SRC) as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)

# ─── Helpers ──────────────────────────────────────────────────────────────────

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
        f"  looking for: {repr(old[:120])}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 0: Version comment
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "V3.2: Fix calib/test score mismatch — ratio-test-only for SIFT/KAZE (no RANSAC)",
    "V3.3: SuperPoint + LightGlue as 4th local component; V3.2.2 calibrated defaults",
    "version comment",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 1: SPECIES_CONFIG — replace with V3.2.2 calibrated defaults + superpoint
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "SPECIES_CONFIG = {\n"
    "    \"SalamanderID2025\": {\n"
    "        # Deformable bodies → SIFT + KAZE + ALIKED (SAM3 100% → masked keypoints)\n"
    "        \"global_weight\": 0.55,\n"
    "        \"local_extractors\": [\"sift\", \"kaze\", \"aliked\"],\n"
    "        \"local_weights\": {\"sift\": 0.15, \"kaze\": 0.10, \"aliked\": 0.20},\n"
    "        \"threshold_known\": 0.40,\n"
    "        \"threshold_cluster\": 0.35,\n"
    "        \"image_size\": 512,\n"
    "        \"qe_k\": 3,\n"
    "    },\n"
    "    \"SeaTurtleID2022\": {\n"
    "        # Rigid, high-contrast features → SIFT + KAZE + ALIKED\n"
    "        \"global_weight\": 0.55,\n"
    "        \"local_extractors\": [\"sift\", \"kaze\", \"aliked\"],\n"
    "        \"local_weights\": {\"sift\": 0.15, \"kaze\": 0.10, \"aliked\": 0.20},\n"
    "        \"threshold_known\": 0.45,\n"
    "        \"threshold_cluster\": 0.40,\n"
    "        \"image_size\": 512,\n"
    "        \"qe_k\": 8,\n"
    "    },\n"
    "    \"LynxID2025\": {\n"
    "        # Rosette/fur patterns → SIFT + KAZE + ALIKED (0% SAM3 → originals)\n"
    "        \"global_weight\": 0.55,\n"
    "        \"local_extractors\": [\"sift\", \"kaze\", \"aliked\"],\n"
    "        \"local_weights\": {\"sift\": 0.15, \"kaze\": 0.10, \"aliked\": 0.20},\n"
    "        \"threshold_known\": 0.40,\n"
    "        \"threshold_cluster\": 0.35,\n"
    "        \"image_size\": 512,\n"
    "        \"qe_k\": 5,\n"
    "    },\n"
    "    \"TexasHornedLizards\": {\n"
    "        # Dense spot patterns → SIFT + KAZE + ALIKED\n"
    "        \"global_weight\": 0.55,\n"
    "        \"local_extractors\": [\"sift\", \"kaze\", \"aliked\"],\n"
    "        \"local_weights\": {\"sift\": 0.15, \"kaze\": 0.10, \"aliked\": 0.20},\n"
    "        \"threshold_known\": None,  # Zero-shot\n"
    "        \"threshold_cluster\": 0.30,\n"
    "        \"image_size\": 512,\n"
    "        \"qe_k\": 5,\n"
    "    },\n"
    "}",
    # ── new SPECIES_CONFIG with V3.2.2 calibrated defaults + superpoint=0.00 ──
    "SPECIES_CONFIG = {\n"
    "    \"SalamanderID2025\": {\n"
    "        # V3.2.2 calibrated: gw=0.50 sw=0.30 kw=0.15 aw=0.05 thr=0.47\n"
    "        \"global_weight\": 0.50,\n"
    "        \"local_extractors\": [\"sift\", \"kaze\", \"aliked\", \"superpoint\"],\n"
    "        \"local_weights\": {\"sift\": 0.30, \"kaze\": 0.15, \"aliked\": 0.05, \"superpoint\": 0.00},\n"
    "        \"threshold_known\": 0.40,\n"
    "        \"threshold_cluster\": 0.47,\n"
    "        \"image_size\": 512,\n"
    "        \"qe_k\": 3,\n"
    "    },\n"
    "    \"SeaTurtleID2022\": {\n"
    "        # V3.2.2 calibrated: gw=0.70 sw=0.20 kw=0.05 aw=0.05 thr=0.57\n"
    "        \"global_weight\": 0.70,\n"
    "        \"local_extractors\": [\"sift\", \"kaze\", \"aliked\", \"superpoint\"],\n"
    "        \"local_weights\": {\"sift\": 0.20, \"kaze\": 0.05, \"aliked\": 0.05, \"superpoint\": 0.00},\n"
    "        \"threshold_known\": 0.45,\n"
    "        \"threshold_cluster\": 0.57,\n"
    "        \"image_size\": 512,\n"
    "        \"qe_k\": 8,\n"
    "    },\n"
    "    \"LynxID2025\": {\n"
    "        # V3.2.2 calibrated: gw=0.40 sw=0.00 kw=0.05 aw=0.55 thr=0.65\n"
    "        \"global_weight\": 0.40,\n"
    "        \"local_extractors\": [\"sift\", \"kaze\", \"aliked\", \"superpoint\"],\n"
    "        \"local_weights\": {\"sift\": 0.00, \"kaze\": 0.05, \"aliked\": 0.55, \"superpoint\": 0.00},\n"
    "        \"threshold_known\": 0.40,\n"
    "        \"threshold_cluster\": 0.65,\n"
    "        \"image_size\": 512,\n"
    "        \"qe_k\": 5,\n"
    "    },\n"
    "    \"TexasHornedLizards\": {\n"
    "        # V3.2.2 calibrated: gw=0.55 sw=0.15 kw=0.10 aw=0.20 thr=0.30 (zero-shot)\n"
    "        \"global_weight\": 0.55,\n"
    "        \"local_extractors\": [\"sift\", \"kaze\", \"aliked\", \"superpoint\"],\n"
    "        \"local_weights\": {\"sift\": 0.15, \"kaze\": 0.10, \"aliked\": 0.20, \"superpoint\": 0.00},\n"
    "        \"threshold_known\": None,  # Zero-shot\n"
    "        \"threshold_cluster\": 0.30,\n"
    "        \"image_size\": 512,\n"
    "        \"qe_k\": 5,\n"
    "    },\n"
    "}",
    "SPECIES_CONFIG with V3.2.2 defaults + superpoint",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 2 (A): Add SuperPointExtractor class after ALIKEDExtractor
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "print(\"\u2713 ALIKED extractor defined (LightGlue, seg_mask filtered)\")",
    "print(\"\u2713 ALIKED extractor defined (LightGlue, seg_mask filtered)\")\n"
    "\n"
    "\n"
    "# \u2500\u2500 V3.3: SuperPoint extractor (LightGlue, 256-dim corner/junction descriptors) \u2500\u2500\n"
    "\n"
    "class SuperPointExtractor:\n"
    "    \"\"\"SuperPoint feature extractor via LightGlue (256-dim, corner/junction-based).\n"
    "\n"
    "    SuperPoint detects repeatable corners and homographic junctions.\n"
    "    This complements ALIKED (texture patches) by targeting structural\n"
    "    keypoints: ear tips, body outlines, limb junctions.\n"
    "    \"\"\"\n"
    "    def __init__(self, max_keypoints=2048, device='cuda'):\n"
    "        from lightglue import SuperPoint as _SP\n"
    "        import os, shutil as _shutil\n"
    "        # Pre-seed torch hub cache to avoid 503 from GitHub releases at runtime\n"
    "        _CKPT = 'superpoint_v1.pth'\n"
    "        _HUB_DIR = os.path.expanduser('~/.cache/torch/hub/checkpoints')\n"
    "        _HUB_PATH = os.path.join(_HUB_DIR, _CKPT)\n"
    "        if not os.path.exists(_HUB_PATH):\n"
    "            for _src in [\n"
    "                f'/kaggle/input/lightglue-weights/{_CKPT}',\n"
    "                f'/kaggle/input/animalclef-v3-3-cache/{_CKPT}',\n"
    "                f'/kaggle/input/animalclef-v3-cache/{_CKPT}',\n"
    "            ]:\n"
    "                if os.path.exists(_src):\n"
    "                    os.makedirs(_HUB_DIR, exist_ok=True)\n"
    "                    _shutil.copy2(_src, _HUB_PATH)\n"
    "                    print(f'  \u2713 SuperPoint weights pre-seeded from {_src}')\n"
    "                    break\n"
    "        self.detector = _SP(max_num_keypoints=max_keypoints).eval().to(device)\n"
    "        self.device = device\n"
    "\n"
    "    def extract(self, image_path):\n"
    "        \"\"\"Extract SuperPoint keypoints, descriptors, and scores.\"\"\"\n"
    "        try:\n"
    "            from lightglue.utils import load_image as _load_sp_img\n"
    "            img = _load_sp_img(str(image_path)).to(self.device)\n"
    "            img_sz = img.shape[-2:]  # (H, W) — for LightGlue image_size\n"
    "            with torch.no_grad():\n"
    "                feats = self.detector.extract(img)\n"
    "            kps    = feats['keypoints'][0].cpu().numpy()\n"
    "            descs  = feats['descriptors'][0].cpu().numpy()   # (N, 256)\n"
    "            scores = feats['keypoint_scores'][0].cpu().numpy()\n"
    "            # Filter keypoints to animal region — same as ALIKEDExtractor\n"
    "            seg_mask = get_seg_mask(str(image_path))\n"
    "            if seg_mask is not None:\n"
    "                h, w = seg_mask.shape[:2]\n"
    "                xs = np.clip(kps[:, 0].astype(int), 0, w - 1)\n"
    "                ys = np.clip(kps[:, 1].astype(int), 0, h - 1)\n"
    "                valid  = seg_mask[ys, xs] > 0\n"
    "                kps    = kps[valid]\n"
    "                descs  = descs[valid]\n"
    "                scores = scores[valid]\n"
    "            if len(kps) < 4:\n"
    "                return None\n"
    "            return {\n"
    "                'keypoints':   kps,\n"
    "                'descriptors': descs,\n"
    "                'scores':      scores,\n"
    "                'image_size':  img_sz,\n"
    "            }\n"
    "        except Exception:\n"
    "            return None\n"
    "\n"
    "print(\"\u2713 SuperPoint extractor defined (LightGlue, seg_mask filtered)\")",
    "Add SuperPointExtractor class",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 3 (C): get_extractor() factory — add 'superpoint' case
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    elif extractor_name == 'aliked':\n"
    "        return ALIKEDExtractor(max_keypoints=1024, device=device)\n"
    "    else:\n"
    "        raise ValueError(f\"Unknown extractor: {extractor_name}. Supported: 'sift', 'kaze', 'aliked'.\")",
    "    elif extractor_name == 'aliked':\n"
    "        return ALIKEDExtractor(max_keypoints=1024, device=device)\n"
    "    elif extractor_name == 'superpoint':\n"
    "        return SuperPointExtractor(max_keypoints=2048, device=device)\n"
    "    else:\n"
    "        raise ValueError(f\"Unknown extractor: {extractor_name}. Supported: 'sift', 'kaze', 'aliked', 'superpoint'.\")",
    "get_extractor factory add superpoint",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 4 (B): Add _get_sp_matcher() + compute_local_score_sp_lg()
#              Insert before compute_pairwise_matches_fast
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "# \u2500\u2500 V3.2: ratio-test-only scoring for SIFT/KAZE (no RANSAC) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "\n"
    "def compute_local_score_ratio_only(feat1, feat2, ratio_thresh=0.75):",
    "# \u2500\u2500 V3.3: SuperPoint LightGlue scorer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "\n"
    "_sp_lg_matcher = None\n"
    "\n"
    "def _get_sp_matcher():\n"
    "    \"\"\"Lazy-init the LightGlue matcher for SuperPoint (loaded on first call).\"\"\"\n"
    "    global _sp_lg_matcher\n"
    "    if _sp_lg_matcher is None:\n"
    "        from lightglue import LightGlue as _LG\n"
    "        _sp_lg_matcher = _LG(features='superpoint').eval().to(DEVICE)\n"
    "    return _sp_lg_matcher\n"
    "\n"
    "\n"
    "def compute_local_score_sp_lg(feat1, feat2):\n"
    "    \"\"\"Score two images using SuperPoint features matched via LightGlue.\n"
    "\n"
    "    Returns: 1 - exp(-n_matches / 20) for consistency with ALIKED scoring.\n"
    "    SuperPoint targets corners/junctions — complements ALIKED texture patches.\n"
    "    \"\"\"\n"
    "    if feat1 is None or feat2 is None:\n"
    "        return 0.0\n"
    "    if len(feat1['keypoints']) < 4 or len(feat2['keypoints']) < 4:\n"
    "        return 0.0\n"
    "\n"
    "    try:\n"
    "        matcher = _get_sp_matcher()\n"
    "        d0 = {\n"
    "            'keypoints':   torch.from_numpy(feat1['keypoints']).float().unsqueeze(0).to(DEVICE),\n"
    "            'descriptors': torch.from_numpy(feat1['descriptors']).float().unsqueeze(0).to(DEVICE),\n"
    "        }\n"
    "        d1 = {\n"
    "            'keypoints':   torch.from_numpy(feat2['keypoints']).float().unsqueeze(0).to(DEVICE),\n"
    "            'descriptors': torch.from_numpy(feat2['descriptors']).float().unsqueeze(0).to(DEVICE),\n"
    "        }\n"
    "        if 'image_size' in feat1:\n"
    "            d0['image_size'] = torch.tensor(feat1['image_size']).float().unsqueeze(0).to(DEVICE)\n"
    "        if 'image_size' in feat2:\n"
    "            d1['image_size'] = torch.tensor(feat2['image_size']).float().unsqueeze(0).to(DEVICE)\n"
    "        with torch.no_grad():\n"
    "            result = matcher({'image0': d0, 'image1': d1})\n"
    "        n_matches = (result['matches0'][0] > -1).sum().item()\n"
    "        return float(1.0 - np.exp(-n_matches / 20.0))\n"
    "    except Exception:\n"
    "        return 0.0\n"
    "\n"
    "\n"
    "# \u2500\u2500 V3.2: ratio-test-only scoring for SIFT/KAZE (no RANSAC) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "\n"
    "def compute_local_score_ratio_only(feat1, feat2, ratio_thresh=0.75):",
    "Add _get_sp_matcher + compute_local_score_sp_lg",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 5 (D): compute_pairwise_matches_fast — cache suffix for superpoint
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    _sfx = \"_lg_matches\" if extractor_type == \"aliked\" else \"_ratio_matches\"\n",
    "    if extractor_type == \"aliked\":\n"
    "        _sfx = \"_lg_matches\"\n"
    "    elif extractor_type == \"superpoint\":\n"
    "        _sfx = \"_sp_lg_matches\"\n"
    "    else:\n"
    "        _sfx = \"_ratio_matches\"\n",
    "cache suffix add superpoint branch",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 6 (D): compute_pairwise_matches_fast — dispatch for superpoint
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "                # V3.2: LightGlue for ALIKED; ratio-test-only for SIFT/KAZE (no RANSAC)\n"
    "                if extractor_type == 'aliked':\n"
    "                    score = compute_local_score_lightglue(features_list[i], features_list[j])\n"
    "                else:\n"
    "                    # No RANSAC gate: score distribution now matches _calib_match_matrix\n"
    "                    score = compute_local_score_ratio_only(features_list[i], features_list[j])",
    "                # V3.3: LightGlue for ALIKED/SuperPoint; ratio-test-only for SIFT/KAZE\n"
    "                if extractor_type == 'aliked':\n"
    "                    score = compute_local_score_lightglue(features_list[i], features_list[j])\n"
    "                elif extractor_type == 'superpoint':\n"
    "                    score = compute_local_score_sp_lg(features_list[i], features_list[j])\n"
    "                else:\n"
    "                    # No RANSAC gate: score distribution now matches _calib_match_matrix\n"
    "                    score = compute_local_score_ratio_only(features_list[i], features_list[j])",
    "dispatch add superpoint branch",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 7: Cell 16 print updated for V3.3
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "print(\"\u2713 Using SIFT + KAZE + ALIKED (LightGlue) + MiewID v3 ensemble (V3.2)\")",
    "print(\"\u2713 Using SIFT + KAZE + ALIKED + SuperPoint (LightGlue) + MiewID v3 ensemble (V3.3)\")",
    "Cell 16 print V3.3",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 8: Two-part cache fix:
#   8a) V3 cache block: also load match_scores/ (SIFT/KAZE/ALIKED from V3.2
#       are valid for V3.3; skip SP files which don't exist in V3 cache).
#   8b) V3.3 cache block: comprehensive — load ALL missing files from
#       local_features/, match_scores/, calib_cache/. After first V3.3 run,
#       user exports all computed files to animalclef-v3-3-cache; subsequent
#       runs load everything (ALIKED pkl, SIFT/KAZE/ALIKED/SP match scores,
#       SP calib, SP pkl) without recomputation.
# ═══════════════════════════════════════════════════════════════════════════════

# ── 8a: Add match_scores loading to the existing V3 cache block ──────────────
# SIFT _ratio_matches and ALIKED _lg_matches from V3.2 are identical in V3.3.
# V3 cache block currently copies embeddings + SIFT/KAZE pkl + calib_cache.
# Add match_scores/ loading (skip _sp_lg_matches which don't exist in V3 cache).
patch_cell(
    nb,
    "    print(f\"\u2713 V3 cache preloaded ({_copied} new files)\")\n"
    "else:\n"
    "    print(\"\u26a0 V3 cache not mounted \u2014 will recompute ALIKED features + all match matrices\")",
    "    # match_scores: _ratio_matches (SIFT/KAZE) + _lg_matches (ALIKED) from V3 cache.\n"
    "    # V3.2 and V3.2.2 (score 0.47912) both used these exact files — they are valid.\n"
    "    # Only _sp_lg_matches are skipped (SP is new in V3.3, not in V3 cache).\n"
    "    _ms_src = os.path.join(_V3_SRC, \"cache\", \"match_scores\")\n"
    "    _ms_dst = os.path.join(\"cache\", \"match_scores\")\n"
    "    os.makedirs(_ms_dst, exist_ok=True)\n"
    "    if os.path.isdir(_ms_src):\n"
    "        for _fname in sorted(os.listdir(_ms_src)):\n"
    "            if \"_sp_lg_matches\" in _fname:\n"
    "                continue  # SP match scores not in V3 cache\n"
    "            _src_f = os.path.join(_ms_src, _fname)\n"
    "            _dst_f = os.path.join(_ms_dst, _fname)\n"
    "            if not os.path.exists(_dst_f):\n"
    "                shutil.copy2(_src_f, _dst_f); _copied += 1\n"
    "                print(f\"  Copied  match_scores/{_fname}\")\n"
    "            else:\n"
    "                print(f\"  Present match_scores/{_fname}\")\n"
    "\n"
    "    print(f\"\u2713 V3 cache preloaded ({_copied} new files)\")\n"
    "else:\n"
    "    print(\"\u26a0 V3 cache not mounted \u2014 will recompute ALIKED features + all match matrices\")",
    "V3 cache block add match_scores loading",
)

# ── 8b: V3.3 cache block — comprehensive (all missing files) ─────────────────
_V33_CACHE_BLOCK = (
    "\n"
    "\n"
    "# \u2500\u2500 V3.3 cache: all V3.3-computed artifacts (local_features + match_scores + calib) \u2500\u2500\n"
    "# After the first V3.3 Kaggle run, export the following to 'animalclef-v3-3-cache':\n"
    "#   cache/local_features/*_aliked.pkl    (seg_mask-filtered; also valid for V3.4+)\n"
    "#   cache/local_features/*_superpoint.pkl\n"
    "#   cache/match_scores/*_lg_matches.npy  (ALIKED)\n"
    "#   cache/match_scores/*_ratio_matches.npy (SIFT, KAZE)\n"
    "#   cache/match_scores/*_sp_lg_matches.npy (SuperPoint)\n"
    "#   calib_cache/*_sp_matrix.npy\n"
    "# Then add 'sreevaatsavbavana/animalclef-v3-3-cache' to dataset_sources.\n"
    "_V33_CANDIDATES = [\n"
    "    \"/kaggle/input/datasets/sreevaatsavbavana/animalclef-v3-3-cache\",\n"
    "    \"/kaggle/input/animalclef-v3-3-cache\",\n"
    "]\n"
    "_V33_SRC = None\n"
    "for _p in _V33_CANDIDATES:\n"
    "    if os.path.exists(_p):\n"
    "        _V33_SRC = _p\n"
    "        break\n"
    "\n"
    "if _V33_SRC:\n"
    "    print(f\"\\nV3.3 cache found: {_V33_SRC}\")\n"
    "    _copied = 0\n"
    "\n"
    "    # Load ALL missing files from local_features, match_scores, calib_cache.\n"
    "    # The if-not-exists guard prevents overwriting files already loaded by\n"
    "    # the V2.8 or V3 cache blocks above.\n"
    "    for _sub, _dst in [\n"
    "        (os.path.join(_V33_SRC, \"cache\", \"local_features\"),\n"
    "         os.path.join(\"cache\", \"local_features\")),\n"
    "        (os.path.join(_V33_SRC, \"cache\", \"match_scores\"),\n"
    "         os.path.join(\"cache\", \"match_scores\")),\n"
    "        (os.path.join(_V33_SRC, \"calib_cache\"),\n"
    "         \"/kaggle/working/calib_cache\"),\n"
    "    ]:\n"
    "        os.makedirs(_dst, exist_ok=True)\n"
    "        if not os.path.isdir(_sub):\n"
    "            continue\n"
    "        for _fname in sorted(os.listdir(_sub)):\n"
    "            _src_f = os.path.join(_sub, _fname)\n"
    "            _dst_f = os.path.join(_dst, _fname)\n"
    "            if not os.path.exists(_dst_f):\n"
    "                shutil.copy2(_src_f, _dst_f); _copied += 1\n"
    "                print(f\"  Copied  {_fname}\")\n"
    "            else:\n"
    "                print(f\"  Present {_fname}\")\n"
    "\n"
    "    print(f\"\u2713 V3.3 cache preloaded ({_copied} new files)\")\n"
    "else:\n"
    "    print(\"\u26a0 V3.3 cache not mounted \u2014 ALIKED + SP features computed at runtime\")"
)

patch_cell(
    nb,
    "else:\n"
    "    print(\"\u26a0 V3 cache not mounted \u2014 will recompute ALIKED features + all match matrices\")",
    "else:\n"
    "    print(\"\u26a0 V3 cache not mounted \u2014 will recompute ALIKED features + all match matrices\")"
    + _V33_CACHE_BLOCK,
    "Add V3.3 cache preload block",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 9: Calibration title
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "V3.2 -- Joint Weight + Threshold Calibration (Training Identities)",
    "V3.3 -- Joint Weight + Threshold Calibration (Training Identities)",
    "calibration title V3.3",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 9b: Rename calibration step labels [x/5] -> [x/6] (V3.3 has 6 steps)
# patch_cell replaces only the FIRST occurrence in each cell, so each label
# variant (loaded / extracting) needs its own call.
# ═══════════════════════════════════════════════════════════════════════════════
for _old_step, _new_step in [
    ('[1/5] Global embeddings: loaded from cache',  '[1/6] Global embeddings: loaded from cache'),
    ('[1/5] Extracting global embeddings...',        '[1/6] Extracting global embeddings...'),
    ('[2/5] SIFT match matrix: loaded from cache',  '[2/6] SIFT match matrix: loaded from cache'),
    ('[2/5] Extracting SIFT descriptors + BFMatcher...', '[2/6] Extracting SIFT descriptors + BFMatcher...'),
    ('[3/5] KAZE match matrix: loaded from cache',  '[3/6] KAZE match matrix: loaded from cache'),
    ('[3/5] Extracting KAZE descriptors + BFMatcher...', '[3/6] Extracting KAZE descriptors + BFMatcher...'),
    ('[4/5] ALIKED matrix: loaded from cache',      '[4/6] ALIKED matrix: loaded from cache'),
    ('[4/5] Extracting ALIKED features + LightGlue matching...', '[4/6] Extracting ALIKED features + LightGlue matching...'),
]:
    patch_cell(nb, _old_step, _new_step, f"step label {_old_step[:6]} -> {_new_step[:6]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 9c: Add CALIB_LG_TOPK = 10 for LightGlue matchers (ALIKED + SP).
#           CALIB_BFM_TOPK=50 is fine for fast BFMatcher (SIFT/KAZE).
#           LightGlue with 50 neighbors × 2957 Lynx images = 74K pairs × ~15ms
#           = 18 hours per component. CALIB_LG_TOPK=10 → ~4 min per component.
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "CALIB_BFM_TOPK  = 50    # top-K preselection per image for BFMatcher",
    "CALIB_BFM_TOPK  = 50    # top-K preselection per image for BFMatcher (SIFT/KAZE)\n"
    "CALIB_LG_TOPK   = 10    # top-K for LightGlue matchers (ALIKED + SP); LG is ~50x slower than BFM",
    "add CALIB_LG_TOPK constant",
)

# Switch ALIKED calibration matrix call to use CALIB_LG_TOPK
patch_cell(
    nb,
    "_calib_match_matrix_lightglue(aliked_feats, global_embs, CALIB_BFM_TOPK, DEVICE)",
    "_calib_match_matrix_lightglue(aliked_feats, global_embs, CALIB_LG_TOPK, DEVICE)",
    "ALIKED calib uses CALIB_LG_TOPK",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Patch 9d: Test-time K for LightGlue (ALIKED/SP) → 20, not 50.
#           K=50 with 2048 keypoints (SP test-time) = 23,650 pairs × ~400ms
#           = 2.6h for Lynx alone. K=20 → 63 min for Lynx SP, ~2.5h total.
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "\n    K = min(50, n)\n",
    "\n    # LightGlue is much slower than BFMatcher — use smaller K\n"
    "    K = min(20, n) if extractor_type in ('aliked', 'superpoint') else min(50, n)\n",
    "test-time LG uses K=20 not K=50",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 10 (F): Calibration grid variables — extended ranges, KAZE explicit,
#               THR_GRID extended to 0.75, SuperPoint as residual
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "GLOBAL_W_GRID   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]\n"
    "SIFT_W_GRID     = [0.0, 0.1, 0.2, 0.3]\n"
    "ALIKED_W_GRID   = [0.0, 0.1, 0.2, 0.3]\n"
    "THR_GRID        = [round(0.15 + i * 0.02, 2) for i in range(23)]  # 0.15..0.59",
    "# V3.3: KAZE explicit (not residual); SIFT/ALIKED ceilings raised;\n"
    "# THR extended to 0.75 (bakes in V3.2.2 live fix); spw = residual.\n"
    "GLOBAL_W_GRID   = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]                        # 7\n"
    "SIFT_W_GRID     = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]            # 9 (raised ceiling)\n"
    "KAZE_W_GRID     = [0.00, 0.05, 0.10, 0.15, 0.20]                                    # 5 (explicit)\n"
    "ALIKED_W_GRID   = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60]            # 9 (covers Lynx aw=0.55)\n"
    "THR_GRID        = np.linspace(0.15, 0.75, 31)                                        # 31 steps (raised ceiling)",
    "calibration grid variables V3.3",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 11: Calibration grid print line (add KAZE_W_GRID)
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "print(f\"  Grid: gw={GLOBAL_W_GRID}  sw={SIFT_W_GRID}  aw={ALIKED_W_GRID}\")\n"
    "print(f\"  Thresholds: {THR_GRID[0]:.2f} .. {THR_GRID[-1]:.2f} ({len(THR_GRID)} steps)\")",
    "print(f\"  Grid: gw={GLOBAL_W_GRID}  sw={SIFT_W_GRID}  kw={KAZE_W_GRID}  aw={ALIKED_W_GRID}\")\n"
    "print(f\"  Thresholds: {THR_GRID[0]:.2f} .. {THR_GRID[-1]:.2f} ({len(THR_GRID)} steps)\")\n"
    "print(f\"  Strategy: gw+sw+kw+aw explicit; spw = residual (=0 if no SuperPoint needed)\")",
    "calibration grid print V3.3",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 12: Add SuperPoint calibration helpers + SP step in loop
#           Insert after _calib_match_matrix_lightglue definition
#           (before "# ── Main calibration loop")
# ═══════════════════════════════════════════════════════════════════════════════
_SP_CALIB_CODE = (
    "\n"
    "\n"
    "# \u2500\u2500 V3.3 Helper 6: SuperPoint feature extraction (GPU) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "def _extract_calib_sp(img_paths, img_keys, device):\n"
    "    '''Extract SuperPoint features for calibration images.\n"
    "\n"
    "    Mirrors _extract_calib_aliked() exactly, using SuperPoint instead.\n"
    "    '''\n"
    "    from lightglue import SuperPoint as _SP\n"
    "    from lightglue.utils import load_image as _li\n"
    "    import os as _os, shutil as _shutil\n"
    "\n"
    "    # Pre-seed torch hub cache (same as SuperPointExtractor.__init__)\n"
    "    _CKPT = 'superpoint_v1.pth'\n"
    "    _HUB_DIR = _os.path.expanduser('~/.cache/torch/hub/checkpoints')\n"
    "    _HUB_PATH = _os.path.join(_HUB_DIR, _CKPT)\n"
    "    if not _os.path.exists(_HUB_PATH):\n"
    "        for _src in [\n"
    "            f'/kaggle/input/lightglue-weights/{_CKPT}',\n"
    "            f'/kaggle/input/animalclef-v3-3-cache/{_CKPT}',\n"
    "            f'/kaggle/input/animalclef-v3-cache/{_CKPT}',\n"
    "        ]:\n"
    "            if _os.path.exists(_src):\n"
    "                _os.makedirs(_HUB_DIR, exist_ok=True)\n"
    "                _shutil.copy2(_src, _HUB_PATH)\n"
    "                break\n"
    "\n"
    "    model = _SP(max_num_keypoints=CALIB_KPT_CAP).eval().to(device)\n"
    "    all_feats = []\n"
    "    for path, key in zip(img_paths, img_keys):\n"
    "        try:\n"
    "            img = _li(str(path)).to(device)\n"
    "            img_sz = img.shape[-2:]  # (H, W)\n"
    "            with torch.no_grad():\n"
    "                feats = model.extract(img)\n"
    "            kps    = feats['keypoints'][0].cpu().numpy()\n"
    "            descs  = feats['descriptors'][0].cpu().numpy()\n"
    "\n"
    "            # Filter by seg_mask (same as ALIKED calibration)\n"
    "            seg_mask = get_seg_mask(path)\n"
    "            if seg_mask is not None:\n"
    "                h, w = seg_mask.shape[:2]\n"
    "                xs = np.clip(kps[:, 0].astype(int), 0, w - 1)\n"
    "                ys = np.clip(kps[:, 1].astype(int), 0, h - 1)\n"
    "                valid = seg_mask[ys, xs] > 0\n"
    "                kps   = kps[valid]\n"
    "                descs = descs[valid]\n"
    "\n"
    "            if len(kps) < 4:\n"
    "                all_feats.append(None)\n"
    "                continue\n"
    "\n"
    "            all_feats.append({\n"
    "                'keypoints':  kps,\n"
    "                'descriptors': descs,\n"
    "                'image_size': img_sz,\n"
    "            })\n"
    "        except Exception:\n"
    "            all_feats.append(None)\n"
    "    del model\n"
    "    torch.cuda.empty_cache()\n"
    "    return all_feats\n"
    "\n"
    "\n"
    "# \u2500\u2500 V3.3 Helper 7: LightGlue match matrix for SuperPoint \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "def _calib_match_matrix_sp_lg(feat_list, global_embs, top_k, device):\n"
    "    '''Build (N, N) match matrix via LightGlue for SuperPoint features.\n"
    "\n"
    "    Identical to _calib_match_matrix_lightglue() but uses features='superpoint'.\n"
    "    '''\n"
    "    from lightglue import LightGlue as _LG\n"
    "\n"
    "    N = len(feat_list)\n"
    "    matrix = np.zeros((N, N), dtype=np.float32)\n"
    "    np.fill_diagonal(matrix, 1.0)\n"
    "\n"
    "    matcher = _LG(features='superpoint').eval().to(device)\n"
    "\n"
    "    g_sim = np.clip(global_embs @ global_embs.T, 0.0, 1.0)\n"
    "    pairs = set()\n"
    "    for i in range(N):\n"
    "        sims_i    = g_sim[i].copy()\n"
    "        sims_i[i] = -1.0\n"
    "        top_j     = np.argsort(sims_i)[-top_k:]\n"
    "        for j in top_j:\n"
    "            pairs.add((min(i, j), max(i, j)))\n"
    "\n"
    "    for i, j in pairs:\n"
    "        fi, fj = feat_list[i], feat_list[j]\n"
    "        if fi is None or fj is None:\n"
    "            continue\n"
    "        try:\n"
    "            d0 = {\n"
    "                'keypoints':   torch.from_numpy(fi['keypoints']).float().unsqueeze(0).to(device),\n"
    "                'descriptors': torch.from_numpy(fi['descriptors']).float().unsqueeze(0).to(device),\n"
    "                'image_size':  torch.tensor(fi['image_size']).float().unsqueeze(0).to(device),\n"
    "            }\n"
    "            d1 = {\n"
    "                'keypoints':   torch.from_numpy(fj['keypoints']).float().unsqueeze(0).to(device),\n"
    "                'descriptors': torch.from_numpy(fj['descriptors']).float().unsqueeze(0).to(device),\n"
    "                'image_size':  torch.tensor(fj['image_size']).float().unsqueeze(0).to(device),\n"
    "            }\n"
    "            with torch.no_grad():\n"
    "                result = matcher({'image0': d0, 'image1': d1})\n"
    "            n_matches    = (result['matches0'][0] > -1).sum().item()\n"
    "            score        = 1.0 - np.exp(-n_matches / 20.0)\n"
    "            matrix[i, j] = score\n"
    "            matrix[j, i] = score\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "    del matcher\n"
    "    torch.cuda.empty_cache()\n"
    "    return matrix\n"
)

patch_cell(
    nb,
    "# \u2500\u2500 Main calibration loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\ncalib_model = get_global_model()",
    _SP_CALIB_CODE
    + "# \u2500\u2500 Main calibration loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\ncalib_model = get_global_model()",
    "Add SP calibration helpers before main loop",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 13: Add SP extraction step (Step 5) in calibration loop
#           Insert after Step 4 (ALIKED extraction block)
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    # Step 5: grid search\n"
    "    from scipy.cluster.hierarchy import linkage as _sp_linkage, fcluster as _sp_fcluster",
    "    # Step 5: SuperPoint match matrix (LightGlue, cached)\n"
    "    _spf = Path(f\"{_cp}_sp_matrix.npy\")\n"
    "    if _spf.exists():\n"
    "        print(f\"  [5/6] SP matrix: loaded from cache\")\n"
    "        sp_matrix = np.load(_spf)\n"
    "    else:\n"
    "        print(f\"  [5/6] Extracting SuperPoint features + LightGlue matching...\")\n"
    "        sp_feats  = _extract_calib_sp(img_paths, img_keys, DEVICE)\n"
    "        sp_matrix = _calib_match_matrix_sp_lg(sp_feats, global_embs, CALIB_LG_TOPK, DEVICE)\n"
    "        np.save(_spf, sp_matrix)\n"
    "\n"
    "    # Step 6: grid search\n"
    "    from scipy.cluster.hierarchy import linkage as _sp_linkage, fcluster as _sp_fcluster",
    "Add SP step 5 in calibration loop",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 14: Update n_combos computation (add kw loop; spw = residual)
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    n_combos = sum(\n"
    "        1 for gw in GLOBAL_W_GRID for sw in SIFT_W_GRID\n"
    "        for aw in ALIKED_W_GRID if round(1.0 - gw - sw - aw, 4) >= 0\n"
    "    )\n"
    "    print(f\"  [5/5] Grid search ({n_combos} weight combos x {len(THR_GRID)} thresholds)...\")",
    "    n_combos = sum(\n"
    "        1 for gw in GLOBAL_W_GRID for sw in SIFT_W_GRID\n"
    "        for kw in KAZE_W_GRID for aw in ALIKED_W_GRID\n"
    "        if round(1.0 - gw - sw - kw - aw, 4) >= 0\n"
    "    )\n"
    "    print(f\"  [6/6] Grid search ({n_combos} weight combos x {len(THR_GRID)} thresholds)...\")",
    "n_combos update for 4-loop grid",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 15: Add best_spw to best_* initialization
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    best_gw  = cfg[\"global_weight\"]\n"
    "    best_sw  = cfg[\"local_weights\"][\"sift\"]\n"
    "    best_kw  = cfg[\"local_weights\"].get(\"kaze\", 0.0)\n"
    "    best_aw  = cfg[\"local_weights\"].get(\"aliked\", 0.0)\n"
    "    best_thr = cfg[\"threshold_cluster\"]\n"
    "    best_ami = -1.0",
    "    best_gw  = cfg[\"global_weight\"]\n"
    "    best_sw  = cfg[\"local_weights\"][\"sift\"]\n"
    "    best_kw  = cfg[\"local_weights\"].get(\"kaze\", 0.0)\n"
    "    best_aw  = cfg[\"local_weights\"].get(\"aliked\", 0.0)\n"
    "    best_spw = cfg[\"local_weights\"].get(\"superpoint\", 0.0)\n"
    "    best_thr = cfg[\"threshold_cluster\"]\n"
    "    best_ami = -1.0",
    "Add best_spw to init",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 16: Replace grid search loop (3 loops → 4 loops; spw = residual)
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
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
    "                        best_ami, best_gw, best_sw, best_kw, best_aw, best_thr = ami, gw, sw, kw, aw, thr",
    "    for gw in GLOBAL_W_GRID:\n"
    "        for sw in SIFT_W_GRID:\n"
    "            for kw in KAZE_W_GRID:\n"
    "                for aw in ALIKED_W_GRID:\n"
    "                    spw = round(1.0 - gw - sw - kw - aw, 4)\n"
    "                    if spw < 0:\n"
    "                        continue\n"
    "                    ensemble = (gw * global_sim + sw * sift_matrix\n"
    "                                + kw * kaze_matrix + aw * aliked_matrix\n"
    "                                + spw * sp_matrix)\n"
    "                    dist_sq = np.clip(1.0 - ensemble, 0.0, 1.0).astype(np.float64)\n"
    "                    # Compute linkage ONCE per weight combo; cut at all thresholds (~0s each)\n"
    "                    Z = _sp_linkage(dist_sq[_triu_idx], method=\"average\")\n"
    "                    for thr in THR_GRID:\n"
    "                        pred = _sp_fcluster(Z, t=thr, criterion=\"distance\")\n"
    "                        ami = _ami_score(true_labels, pred)\n"
    "                        if ami > best_ami:\n"
    "                            best_ami, best_gw, best_sw, best_kw, best_aw, best_spw, best_thr = ami, gw, sw, kw, aw, spw, thr",
    "Replace grid search 3-loop with 4-loop + spw residual",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 17: Update calibrated_config write (add sp_weight)
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    calibrated_config[sp] = {\n"
    "        \"global_weight\":  best_gw,\n"
    "        \"sift_weight\":    best_sw,\n"
    "        \"kaze_weight\":    best_kw,\n"
    "        \"aliked_weight\":  best_aw,\n"
    "        \"threshold\":      best_thr,\n"
    "        \"ami\":            best_ami,\n"
    "    }",
    "    calibrated_config[sp] = {\n"
    "        \"global_weight\":  best_gw,\n"
    "        \"sift_weight\":    best_sw,\n"
    "        \"kaze_weight\":    best_kw,\n"
    "        \"aliked_weight\":  best_aw,\n"
    "        \"sp_weight\":      best_spw,\n"
    "        \"threshold\":      best_thr,\n"
    "        \"ami\":            best_ami,\n"
    "    }",
    "Add sp_weight to calibrated_config",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 18: Update prev_*/best_* print to include spw
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    prev_gw  = cfg[\"global_weight\"]\n"
    "    prev_sw  = cfg[\"local_weights\"][\"sift\"]\n"
    "    prev_kw  = cfg[\"local_weights\"].get(\"kaze\", 0.0)\n"
    "    prev_aw  = cfg[\"local_weights\"].get(\"aliked\", 0.0)\n"
    "    prev_thr = cfg[\"threshold_cluster\"]\n"
    "    print(f\"  weights:   ({prev_gw:.2f}, {prev_sw:.2f}, {prev_kw:.2f}, {prev_aw:.2f})\"\n"
    "          f\" -> ({best_gw:.2f}, {best_sw:.2f}, {best_kw:.2f}, {best_aw:.2f})\")\n"
    "    print(f\"  threshold: {prev_thr:.2f} -> {best_thr:.2f}\"\n"
    "          f\"  (AMI={best_ami:.4f}, t={_time.time()-t0:.1f}s)\")",
    "    prev_gw  = cfg[\"global_weight\"]\n"
    "    prev_sw  = cfg[\"local_weights\"][\"sift\"]\n"
    "    prev_kw  = cfg[\"local_weights\"].get(\"kaze\", 0.0)\n"
    "    prev_aw  = cfg[\"local_weights\"].get(\"aliked\", 0.0)\n"
    "    prev_spw = cfg[\"local_weights\"].get(\"superpoint\", 0.0)\n"
    "    prev_thr = cfg[\"threshold_cluster\"]\n"
    "    print(f\"  weights:   ({prev_gw:.2f}, {prev_sw:.2f}, {prev_kw:.2f}, {prev_aw:.2f}, {prev_spw:.2f})\"\n"
    "          f\" -> ({best_gw:.2f}, {best_sw:.2f}, {best_kw:.2f}, {best_aw:.2f}, {best_spw:.2f})\")\n"
    "    print(f\"  threshold: {prev_thr:.2f} -> {best_thr:.2f}\"\n"
    "          f\"  (AMI={best_ami:.4f}, t={_time.time()-t0:.1f}s)\")",
    "Add prev_spw and best_spw to weights print",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 19: Update THL calibrated_config to include sp_weight
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "calibrated_config[\"TexasHornedLizards\"] = {\n"
    "    \"global_weight\":  thl_cfg[\"global_weight\"],\n"
    "    \"sift_weight\":    thl_cfg[\"local_weights\"][\"sift\"],\n"
    "    \"kaze_weight\":    thl_cfg[\"local_weights\"][\"kaze\"],\n"
    "    \"aliked_weight\":  thl_cfg[\"local_weights\"][\"aliked\"],\n"
    "    \"threshold\":      thl_cfg[\"threshold_cluster\"],\n"
    "    \"ami\":            None,\n"
    "}",
    "calibrated_config[\"TexasHornedLizards\"] = {\n"
    "    \"global_weight\":  thl_cfg[\"global_weight\"],\n"
    "    \"sift_weight\":    thl_cfg[\"local_weights\"][\"sift\"],\n"
    "    \"kaze_weight\":    thl_cfg[\"local_weights\"][\"kaze\"],\n"
    "    \"aliked_weight\":  thl_cfg[\"local_weights\"][\"aliked\"],\n"
    "    \"sp_weight\":      thl_cfg[\"local_weights\"].get(\"superpoint\", 0.0),\n"
    "    \"threshold\":      thl_cfg[\"threshold_cluster\"],\n"
    "    \"ami\":            None,\n"
    "}",
    "Add sp_weight to THL calibrated_config",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 20 (H): SPECIES_CONFIG write-back — add superpoint key
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    if \"aliked\" in SPECIES_CONFIG[sp][\"local_weights\"]:\n"
    "        SPECIES_CONFIG[sp][\"local_weights\"][\"aliked\"] = cal[\"aliked_weight\"]\n"
    "    SPECIES_CONFIG[sp][\"threshold_cluster\"]     = cal[\"threshold\"]",
    "    if \"aliked\" in SPECIES_CONFIG[sp][\"local_weights\"]:\n"
    "        SPECIES_CONFIG[sp][\"local_weights\"][\"aliked\"] = cal[\"aliked_weight\"]\n"
    "    if \"superpoint\" in SPECIES_CONFIG[sp][\"local_weights\"]:\n"
    "        SPECIES_CONFIG[sp][\"local_weights\"][\"superpoint\"] = cal[\"sp_weight\"]\n"
    "    SPECIES_CONFIG[sp][\"threshold_cluster\"]     = cal[\"threshold\"]",
    "Write-back add superpoint",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 21: Final calibrated SPECIES_CONFIG print — add spw column
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "    print(f\"  {sp:<25}: gw={cfg['global_weight']:.2f}\"\n"
    "          f\"  sw={lw.get('sift', 0):.2f}\"\n"
    "          f\"  kw={lw.get('kaze', 0):.2f}\"\n"
    "          f\"  aw={lw.get('aliked', 0):.2f}\"\n"
    "          f\"  thr={cfg['threshold_cluster']:.2f}\")",
    "    print(f\"  {sp:<25}: gw={cfg['global_weight']:.2f}\"\n"
    "          f\"  sw={lw.get('sift', 0):.2f}\"\n"
    "          f\"  kw={lw.get('kaze', 0):.2f}\"\n"
    "          f\"  aw={lw.get('aliked', 0):.2f}\"\n"
    "          f\"  spw={lw.get('superpoint', 0):.2f}\"\n"
    "          f\"  thr={cfg['threshold_cluster']:.2f}\")",
    "Add spw to final calibrated config print",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Write output
# ═══════════════════════════════════════════════════════════════════════════════
with open(DST, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {DST}  ({len(nb['cells'])} cells)")


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════
print("\nVerification:")

with open(DST) as f:
    out_text = f.read()

nb_out = json.loads(out_text)
all_src = ""
for cell in nb_out["cells"]:
    s = cell["source"]
    all_src += (s if isinstance(s, str) else "".join(s)) + "\n"

checks = [
    # ── Patch 0: version comment ──
    ("version comment V3.3",
     "V3.3: SuperPoint + LightGlue as 4th local component"),
    ("OLD V3.2 version comment gone",
     "V3.2: Fix calib/test score mismatch"),

    # ── Patch 1: SPECIES_CONFIG V3.2.2 defaults ──
    ("Lynx gw=0.40 in config",
     '"global_weight": 0.40,'),
    ("Lynx aw=0.55 in config",
     '"aliked": 0.55, "superpoint": 0.00'),
    ("Lynx thr=0.65 in config",
     '"threshold_cluster": 0.65,'),
    ("Sal gw=0.50 in config",
     '"sift": 0.30, "kaze": 0.15, "aliked": 0.05, "superpoint": 0.00'),
    ("Sal thr=0.47 in config",
     '"threshold_cluster": 0.47,'),
    ("Turtle gw=0.70 in config",
     '"sift": 0.20, "kaze": 0.05, "aliked": 0.05, "superpoint": 0.00'),
    ("Turtle thr=0.57 in config",
     '"threshold_cluster": 0.57,'),
    ("THL thr=0.30 in config",
     '"sift": 0.15, "kaze": 0.10, "aliked": 0.20, "superpoint": 0.00'),
    ("superpoint in local_extractors",
     '"superpoint"'),
    ("OLD default weights gone",
     '"sift": 0.15, "kaze": 0.10, "aliked": 0.20},\n        "threshold_known": 0.40'),

    # ── Patch 2 (A): SuperPointExtractor ──
    ("SuperPointExtractor class defined",
     "class SuperPointExtractor:"),
    ("SuperPoint max_num_keypoints=2048",
     "_SP(max_num_keypoints=max_keypoints).eval()"),
    ("SuperPoint hub cache pre-seed",
     "superpoint_v1.pth"),
    ("SuperPoint weight candidates list",
     "/kaggle/input/lightglue-weights/"),
    ("SuperPoint descriptor dim comment",
     "# (N, 256)"),
    ("SuperPoint seg_mask filter",
     "# Filter keypoints to animal region — same as ALIKEDExtractor"),
    ("SuperPoint image_size from shape",
     "img_sz = img.shape[-2:]  # (H, W)"),
    ("SuperPoint extractor print",
     "\u2713 SuperPoint extractor defined (LightGlue, seg_mask filtered)"),

    # ── Patch 3 (C): get_extractor factory ──
    ("superpoint in get_extractor",
     "elif extractor_name == 'superpoint':"),
    ("SuperPointExtractor in factory",
     "return SuperPointExtractor(max_keypoints=2048, device=device)"),

    # ── Patch 4 (B): SP matcher + scorer ──
    ("_sp_lg_matcher defined",
     "_sp_lg_matcher = None"),
    ("_get_sp_matcher defined",
     "def _get_sp_matcher():"),
    ("_get_sp_matcher loads superpoint",
     "_LG(features='superpoint').eval()"),
    ("compute_local_score_sp_lg defined",
     "def compute_local_score_sp_lg(feat1, feat2):"),
    ("sp_lg uses matches0",
     "n_matches = (result['matches0'][0] > -1).sum().item()"),
    ("sp_lg score formula",
     "return float(1.0 - np.exp(-n_matches / 20.0))"),

    # ── Patch 5 (D): cache suffix ──
    ("superpoint cache suffix",
     "_sfx = \"_sp_lg_matches\""),
    ("OLD single-line suffix gone",
     "_sfx = \"_lg_matches\" if extractor_type == \"aliked\" else \"_ratio_matches\""),

    # ── Patch 6 (D): dispatch ──
    ("superpoint dispatch added",
     "elif extractor_type == 'superpoint':"),
    ("sp_lg used in dispatch",
     "score = compute_local_score_sp_lg(features_list[i], features_list[j])"),
    ("V3.3 dispatch comment",
     "# V3.3: LightGlue for ALIKED/SuperPoint; ratio-test-only for SIFT/KAZE"),

    # ── Patch 7: Cell 16 print ──
    ("Cell 16 print V3.3",
     "SIFT + KAZE + ALIKED + SuperPoint (LightGlue) + MiewID v3 ensemble (V3.3)"),

    # ── Patch 8: cache blocks ──
    ("V3 cache loads match_scores",
     "_ms_src = os.path.join(_V3_SRC, \"cache\", \"match_scores\")"),
    ("V3 cache match_scores skip SP only",
     "if \"_sp_lg_matches\" in _fname:"),
    ("V3.3 cache candidates",
     "\"/kaggle/input/animalclef-v3-3-cache\""),
    ("V3.3 cache comprehensive loop",
     "os.path.join(_V33_SRC, \"cache\", \"local_features\"),"),
    ("V3.3 cache covers all subdirs",
     "os.path.join(_V33_SRC, \"calib_cache\"),"),
    ("V3.3 cache comment mentions aliked",
     "cache/local_features/*_aliked.pkl"),

    # ── Patch 9c: CALIB_LG_TOPK ──
    ("CALIB_LG_TOPK defined",
     "CALIB_LG_TOPK   = 10"),
    ("ALIKED calib uses LG_TOPK",
     "_calib_match_matrix_lightglue(aliked_feats, global_embs, CALIB_LG_TOPK"),
    ("SP calib uses LG_TOPK",
     "_calib_match_matrix_sp_lg(sp_feats, global_embs, CALIB_LG_TOPK"),
    ("ALIKED calib not using BFM_TOPK",
     "_calib_match_matrix_lightglue(aliked_feats, global_embs, CALIB_BFM_TOPK"),
    ("test-time LG K=20",
     "K = min(20, n) if extractor_type in ('aliked', 'superpoint')"),
    ("old K=min(50) gone",
     "\n    K = min(50, n)\n"),


    # ── Patch 9: calibration title + step labels ──
    ("calib title V3.3",
     "V3.3 -- Joint Weight + Threshold Calibration"),
    ("step 1 renamed to /6",
     "[1/6] Global embeddings"),
    ("step 2 renamed to /6",
     "[2/6] SIFT match matrix"),
    ("step 3 renamed to /6",
     "[3/6] KAZE match matrix"),
    ("step 4 renamed to /6",
     "[4/6] ALIKED matrix"),
    ("old /5 step labels gone",
     "[4/5] ALIKED matrix"),

    # ── Patch 10 (F): calibration grid ──
    ("KAZE_W_GRID defined",
     "KAZE_W_GRID     = [0.00, 0.05, 0.10, 0.15, 0.20]"),
    ("SIFT ceiling 0.40",
     "0.30, 0.35, 0.40]"),
    ("ALIKED covers 0.55/0.60",
     "0.50, 0.55, 0.60]"),
    ("THR_GRID linspace to 0.75",
     "np.linspace(0.15, 0.75, 31)"),
    ("OLD THR_GRID gone",
     "[round(0.15 + i * 0.02, 2) for i in range(23)]"),

    # ── Patch 11: calibration grid print ──
    ("kw in grid print",
     "kw={KAZE_W_GRID}"),
    ("spw strategy note",
     "spw = residual (=0 if no SuperPoint needed)"),

    # ── Patch 12: SP calibration helpers ──
    ("_extract_calib_sp defined",
     "def _extract_calib_sp(img_paths, img_keys, device):"),
    ("_extract_calib_sp hub pre-seed",
     "# Pre-seed torch hub cache (same as SuperPointExtractor.__init__)"),
    ("_calib_match_matrix_sp_lg defined",
     "def _calib_match_matrix_sp_lg(feat_list, global_embs, top_k, device):"),
    ("SP calib uses superpoint features key",
     "_LG(features='superpoint').eval()"),

    # ── Patch 13: SP step in calib loop ──
    ("SP step 5/6 in loop",
     "  [5/6] SP matrix"),
    ("sp_matrix computed in calib",
     "sp_matrix = _calib_match_matrix_sp_lg(sp_feats"),
    ("Grid search is step 6/6",
     "  [6/6] Grid search"),

    # ── Patch 14: n_combos ──
    ("n_combos has kw loop",
     "for kw in KAZE_W_GRID for aw in ALIKED_W_GRID"),
    ("n_combos checks spw >= 0",
     "round(1.0 - gw - sw - kw - aw, 4) >= 0"),

    # ── Patch 15: best_spw init ──
    ("best_spw initialized",
     "best_spw = cfg[\"local_weights\"].get(\"superpoint\", 0.0)"),

    # ── Patch 16: 4-loop grid search ──
    ("4th loop for kw",
     "for kw in KAZE_W_GRID:"),
    ("spw = 1-gw-sw-kw-aw",
     "spw = round(1.0 - gw - sw - kw - aw, 4)"),
    ("spw in ensemble",
     "+ spw * sp_matrix)"),
    ("best update includes spw",
     "best_ami, best_gw, best_sw, best_kw, best_aw, best_spw, best_thr = ami"),

    # ── Patch 17: calibrated_config sp_weight ──
    ("sp_weight in calibrated_config",
     '"sp_weight":      best_spw,'),

    # ── Patch 18: prev_spw print ──
    ("prev_spw in weights print",
     "prev_spw = cfg[\"local_weights\"].get(\"superpoint\", 0.0)"),
    ("best_spw in weights print",
     ", {prev_spw:.2f})\""),

    # ── Patch 19: THL sp_weight ──
    ("THL sp_weight in calibrated_config",
     '"sp_weight":      thl_cfg["local_weights"].get("superpoint", 0.0),'),

    # ── Patch 20 (H): write-back ──
    ("superpoint write-back",
     'SPECIES_CONFIG[sp]["local_weights"]["superpoint"] = cal["sp_weight"]'),

    # ── Patch 21: final print ──
    ("spw in final print",
     'spw={lw.get(\'superpoint\', 0):.2f}'),

    # ── V3.2 structure preserved ──
    ("ratio_only preserved",
     "def compute_local_score_ratio_only(feat1, feat2"),
    ("lightglue install preserved",
     "git+https://github.com/cvg/LightGlue.git"),
    ("ALIKEDExtractor preserved",
     "from lightglue import ALIKED as _ALIKED_Extractor"),
    ("LightGlue scoring preserved",
     "def compute_local_score_lightglue(feat1, feat2):"),
    ("ALIKED seg_mask preserved",
     "ALIKED extractor defined (LightGlue, seg_mask filtered)"),
    ("calib ALIKED step preserved",
     "_calib_match_matrix_lightglue"),
    ("compute_local_score_gpu preserved",
     "def compute_local_score_gpu(feat1, feat2"),
    ("RANSAC code preserved",
     "cv2.findFundamentalMat"),
    ("_ratio_matches suffix preserved",
     '"_ratio_matches"'),
    ("_lg_matches suffix preserved",
     '"_lg_matches"'),
    ("V3.2 ratio-only dispatch preserved",
     "score = compute_local_score_ratio_only(features_list[i], features_list[j])"),
    ("V2.8 yellow mask preserved",
     "if 'SalamanderID2025' in rel_key:"),
    ("scipy linkage preserved",
     "_sp_linkage(dist_sq[_triu_idx], method=\"average\")"),
    ("ensemble cache refresh preserved",
     "ensemble_similarity_cache[_sp] = compute_ensemble_similarity_matrix(_sp)"),
    ("Patch 9 refresh print preserved",
     "ensemble_similarity_cache refreshed"),
    ("del _sys fix preserved",
     "del _sys, _hf_stale\n"),
]

ABSENT = {
    "OLD V3.2 version comment gone",
    "OLD default weights gone",
    "OLD single-line suffix gone",
    "OLD THR_GRID gone",
    "old /5 step labels gone",
    "ALIKED calib not using BFM_TOPK",
    "old K=min(50) gone",
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
              f"{repr(pattern[:80])}")
    else:
        passed += 1
        print(f"  {mark}  {name}")

print(f"\n{passed}/{passed+failed} checks passed")
if failed:
    sys.exit(1)
else:
    print("All checks passed — V3.3 notebook ready for submission.")
