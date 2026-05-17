"""
Build script for V3.2 notebook.

Changes from V3.1:
  1. Fix calibration/test score mismatch for SIFT/KAZE.
     Root cause of V3.1 regression (0.41500 vs V2.8 0.46202):
       - Calibration (_calib_match_matrix): GPU torch.cdist + Lowe ratio test — NO RANSAC.
       - Test V3.1 (compute_local_score_gpu): ratio test + FM_RANSAC gate.
       - RANSAC rejects many matches at test time → test scores systematically lower
         than calibration → calibration grid-search sets kw=0 for all species because
         KAZE/SIFT appear weak at test time (calibration expected higher scores).
     Fix:
       - Add compute_local_score_ratio_only(): GPU ratio test only, no RANSAC.
       - Use for SIFT/KAZE in compute_pairwise_matches_fast() instead of
         compute_local_score_gpu(). Score distributions now match calibration.
       - compute_local_score_gpu() is preserved in notebook (not called for SIFT/KAZE).
  2. Cache suffix renamed: _ransac_matches -> _ratio_matches (forces clean re-compute).
  3. Fix ALIKEDExtractor.extract(): apply get_seg_mask() to filter keypoints to animal
     region — same as _extract_calib_aliked() in calibration. Without this, calibration
     ALIKED features (mask-filtered, clean) differ from test ALIKED features (unfiltered,
     noisy background keypoints) → calibration/test ALIKED score mismatch.
  4. Fix del _k NameError in Cell 2: if _hf_stale is empty, _k is never bound by the
     for-loop, so `del _k` raises NameError. Fix: remove `_k` from the del statement.
  5. Fix stale Cell 16 comment: still says "SIFT-Only Ensemble" from early prototype.
  6. Calibration title updated.
"""

import json, copy, pathlib, sys

SRC = pathlib.Path("FINAL_SOLUTION_v3_1/ensemble_global_local_reid_v3_1.ipynb")
DST = pathlib.Path("FINAL_SOLUTION_v3_2/ensemble_global_local_reid_v3_2.ipynb")

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
        f"  looking for: {repr(old[:80])}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 0: Version comment
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "V3.1: ALIKED + LightGlue as 3rd local component; V3.0 RANSAC for SIFT/KAZE",
    "V3.2: Fix calib/test score mismatch — ratio-test-only for SIFT/KAZE (no RANSAC)",
    "version comment",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 1: Insert compute_local_score_ratio_only before compute_pairwise_matches_fast
# ═══════════════════════════════════════════════════════════════════════════════
# The anchor is at the end of compute_local_score_lightglue (the except block
# `return 0.0` + final return statement) immediately before
# compute_pairwise_matches_fast. This sequence is unique in the notebook.
_RATIO_ANCHOR = (
    "        return 0.0\n"
    "\n"
    "    return float(1.0 - np.exp(-n_matches / 20.0))\n"
    "\n"
    "\n"
    "def compute_pairwise_matches_fast"
)

_RATIO_REPLACEMENT = (
    "        return 0.0\n"
    "\n"
    "    return float(1.0 - np.exp(-n_matches / 20.0))\n"
    "\n"
    "\n"
    "# \u2500\u2500 V3.2: ratio-test-only scoring for SIFT/KAZE (no RANSAC) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "\n"
    "def compute_local_score_ratio_only(feat1, feat2, ratio_thresh=0.75):\n"
    "    \"\"\"GPU Lowe ratio test only \u2014 no RANSAC geometric gate.\n"
    "\n"
    "    V3.2 fix: Calibration (_calib_match_matrix) uses GPU torch.cdist + ratio\n"
    "    test without RANSAC. V3.1 used compute_local_score_gpu at test time which\n"
    "    applies FM_RANSAC, making test scores systematically lower than calibration\n"
    "    scores. Grid-search then incorrectly concluded kw=0 for all species.\n"
    "\n"
    "    This function matches the calibration's score distribution exactly:\n"
    "    Score = 1 \u2212 exp(\u2212n_ratio_matches / 20), same formula as _calib_match_matrix.\n"
    "    \"\"\"\n"
    "    if feat1 is None or feat2 is None:\n"
    "        return 0.0\n"
    "    desc1 = torch.from_numpy(feat1['descriptors']).float().to(DEVICE)\n"
    "    desc2 = torch.from_numpy(feat2['descriptors']).float().to(DEVICE)\n"
    "    n1, n2 = len(desc1), len(desc2)\n"
    "    if n1 < 2 or n2 < 2:\n"
    "        return 0.0\n"
    "\n"
    "    dists = torch.cdist(desc1, desc2, p=2)\n"
    "    k = min(2, n2)\n"
    "    topk_dists, _ = torch.topk(dists, k=k, dim=1, largest=False)\n"
    "    if k == 2:\n"
    "        ratios    = topk_dists[:, 0] / (topk_dists[:, 1] + 1e-8)\n"
    "        n_matches = int((ratios < ratio_thresh).sum().item())\n"
    "    else:\n"
    "        n_matches = n1\n"
    "    return float(1.0 - np.exp(-n_matches / 20.0))\n"
    "\n"
    "\n"
    "def compute_pairwise_matches_fast"
)

patch_cell(nb, _RATIO_ANCHOR, _RATIO_REPLACEMENT, "insert compute_local_score_ratio_only")


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 2: Dispatch — SIFT/KAZE now use compute_local_score_ratio_only
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "                # V3.1: LightGlue for ALIKED, RANSAC gate for SIFT/KAZE\n"
    "                if extractor_type == 'aliked':\n"
    "                    score = compute_local_score_lightglue(features_list[i], features_list[j])\n"
    "                else:\n"
    "                    score = compute_local_score_gpu(features_list[i], features_list[j])",
    "                # V3.2: LightGlue for ALIKED; ratio-test-only for SIFT/KAZE (no RANSAC)\n"
    "                if extractor_type == 'aliked':\n"
    "                    score = compute_local_score_lightglue(features_list[i], features_list[j])\n"
    "                else:\n"
    "                    # No RANSAC gate: score distribution now matches _calib_match_matrix\n"
    "                    score = compute_local_score_ratio_only(features_list[i], features_list[j])",
    "dispatch SIFT/KAZE to ratio-only",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 3: Cache suffix rename (_ransac_matches -> _ratio_matches)
# Forces re-computation with the corrected scoring function.
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    '    _sfx = "_lg_matches" if extractor_type == "aliked" else "_ransac_matches"\n',
    '    _sfx = "_lg_matches" if extractor_type == "aliked" else "_ratio_matches"\n',
    "cache suffix rename",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 4: Calibration title
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "V3.1 -- Joint Weight + Threshold Calibration (Training Identities)",
    "V3.2 -- Joint Weight + Threshold Calibration (Training Identities)",
    "calibration title",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 5: ALIKEDExtractor.extract() — add get_seg_mask() filter
# ═══════════════════════════════════════════════════════════════════════════════
# BUG: _extract_calib_aliked() applies seg_mask to filter ALIKED keypoints to
# the animal region (removing background). ALIKEDExtractor.extract() did NOT,
# so calibration ALIKED features were cleaner than test features → score
# distribution mismatch → aw calibrated higher than test time can deliver.
# Fix: apply the same seg_mask logic in ALIKEDExtractor.extract().
patch_cell(
    nb,
    "            if len(kps) < 4:\n"
    "                return None\n"
    "            return {\n"
    "                'keypoints': kps,\n"
    "                'descriptors': descs,\n"
    "                'scores': scores,\n"
    "                'image_size': img_sz,\n"
    "            }\n"
    "        except Exception:\n"
    "            return None\n"
    "\n"
    "print(\"\u2713 ALIKED extractor defined (LightGlue)\")",
    "            # Filter keypoints to animal region — matches _extract_calib_aliked()\n"
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
    "                'keypoints': kps,\n"
    "                'descriptors': descs,\n"
    "                'scores': scores,\n"
    "                'image_size': img_sz,\n"
    "            }\n"
    "        except Exception:\n"
    "            return None\n"
    "\n"
    "print(\"\u2713 ALIKED extractor defined (LightGlue, seg_mask filtered)\")",
    "ALIKEDExtractor add seg_mask filter",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 6: Fix del _k NameError in Cell 2
# ═══════════════════════════════════════════════════════════════════════════════
# BUG: `_k` is a for-loop variable. If `_hf_stale` is empty (no huggingface_hub
# in sys.modules), the loop body never runs, so `_k` is never bound.
# `del _sys, _hf_stale, _k` then raises NameError: name '_k' is not defined.
# Fix: remove `_k` from the del statement — it doesn't need explicit deletion.
patch_cell(
    nb,
    "del _sys, _hf_stale, _k\n",
    "del _sys, _hf_stale\n",
    "del _k NameError fix",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 7: Fix stale Cell 16 comment ("SIFT-Only Ensemble")
# ═══════════════════════════════════════════════════════════════════════════════
patch_cell(
    nb,
    "print(\"\u2713 Using SIFT + MiewID v3 ensemble (robust, compatible approach)\")",
    "print(\"\u2713 Using SIFT + KAZE + ALIKED (LightGlue) + MiewID v3 ensemble (V3.2)\")",
    "stale Cell 16 comment",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 8: Add V3 cache preloading block (after existing V2.8 block in Cell 6)
# ═══════════════════════════════════════════════════════════════════════════════
# What is valid to reuse from animalclef-v3-cache:
#   cache/embeddings/            — MiewID unchanged ✓
#   cache/local_features/*_{sift|kaze}.pkl  — extractors unchanged ✓
#   calib_cache/*                — ALL 4 matrices valid:
#       _global_embs.npy         — MiewID unchanged ✓
#       _sift_matrix.npy         — _calib_match_matrix uses GPU ratio-only, same in V3.1/V3.2 ✓
#       _kaze_matrix.npy         — same ✓
#       _aliked_matrix.npy       — _extract_calib_aliked already applied seg_mask in V3.1 ✓
# What must NOT be reused:
#   cache/local_features/*_aliked.pkl    — ALIKEDExtractor now applies seg_mask → features differ ✗
#   cache/match_scores/                  — SIFT/KAZE suffix changed (_ransac→_ratio);
#                                          ALIKED match matrices built from stale features ✗
#
# Biggest time save: calib_cache avoids ~2-3h of per-species extraction + LightGlue matching.
_V3_CACHE_BLOCK = (
    "\n"
    "\n"
    "# \u2500\u2500 V3 cache: embeddings + SIFT/KAZE local features + calib_cache \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "# What is reused: embeddings, SIFT/KAZE pkl, calib_cache (all 4 matrices per species).\n"
    "# Skipped: *_aliked.pkl (ALIKEDExtractor now applies seg_mask \u2192 features changed).\n"
    "# Skipped: match_scores/ (SIFT/KAZE suffix _ransac\u2192_ratio; ALIKED features changed).\n"
    "_V3_CANDIDATES = [\n"
    "    \"/kaggle/input/datasets/svrresearch/animalclef-v3-cache\",\n"
    "    \"/kaggle/input/animalclef-v3-cache\",\n"
    "]\n"
    "_V3_SRC = None\n"
    "for _p in _V3_CANDIDATES:\n"
    "    if os.path.exists(_p):\n"
    "        _V3_SRC = _p\n"
    "        break\n"
    "\n"
    "if _V3_SRC:\n"
    "    print(f\"\\nV3 cache found: {_V3_SRC}\")\n"
    "    _copied = 0\n"
    "\n"
    "    # Embeddings (MiewID unchanged \u2014 always valid)\n"
    "    _e_src = os.path.join(_V3_SRC, \"cache\", \"embeddings\")\n"
    "    _e_dst = os.path.join(\"cache\", \"embeddings\")\n"
    "    os.makedirs(_e_dst, exist_ok=True)\n"
    "    if os.path.isdir(_e_src):\n"
    "        for _fname in sorted(os.listdir(_e_src)):\n"
    "            _src_f = os.path.join(_e_src, _fname)\n"
    "            _dst_f = os.path.join(_e_dst, _fname)\n"
    "            if not os.path.exists(_dst_f):\n"
    "                shutil.copy2(_src_f, _dst_f); _copied += 1\n"
    "                print(f\"  Copied  embeddings/{_fname}\")\n"
    "            else:\n"
    "                print(f\"  Present embeddings/{_fname}\")\n"
    "\n"
    "    # SIFT + KAZE local features (extractors unchanged)\n"
    "    # Skip *_aliked.pkl \u2014 ALIKEDExtractor now applies seg_mask (features changed)\n"
    "    _lf_src = os.path.join(_V3_SRC, \"cache\", \"local_features\")\n"
    "    _lf_dst = os.path.join(\"cache\", \"local_features\")\n"
    "    os.makedirs(_lf_dst, exist_ok=True)\n"
    "    if os.path.isdir(_lf_src):\n"
    "        for _fname in sorted(os.listdir(_lf_src)):\n"
    "            if \"_aliked.pkl\" in _fname:\n"
    "                print(f\"  Skipped local_features/{_fname}  (seg_mask fix \u2192 must recompute)\")\n"
    "                continue\n"
    "            _src_f = os.path.join(_lf_src, _fname)\n"
    "            _dst_f = os.path.join(_lf_dst, _fname)\n"
    "            if not os.path.exists(_dst_f):\n"
    "                shutil.copy2(_src_f, _dst_f); _copied += 1\n"
    "                print(f\"  Copied  local_features/{_fname}\")\n"
    "            else:\n"
    "                print(f\"  Present local_features/{_fname}\")\n"
    "\n"
    "    # calib_cache \u2014 ALL matrices valid (see comment above)\n"
    "    _cc_src = os.path.join(_V3_SRC, \"calib_cache\")\n"
    "    _cc_dst = \"/kaggle/working/calib_cache\"\n"
    "    os.makedirs(_cc_dst, exist_ok=True)\n"
    "    if os.path.isdir(_cc_src):\n"
    "        for _fname in sorted(os.listdir(_cc_src)):\n"
    "            _src_f = os.path.join(_cc_src, _fname)\n"
    "            _dst_f = os.path.join(_cc_dst, _fname)\n"
    "            if not os.path.exists(_dst_f):\n"
    "                shutil.copy2(_src_f, _dst_f); _copied += 1\n"
    "                print(f\"  Copied  calib_cache/{_fname}\")\n"
    "            else:\n"
    "                print(f\"  Present calib_cache/{_fname}\")\n"
    "\n"
    "    print(f\"\u2713 V3 cache preloaded ({_copied} new files)\")\n"
    "else:\n"
    "    print(\"\u26a0 V3 cache not mounted \u2014 will recompute ALIKED features + all match matrices\")"
)

patch_cell(
    nb,
    "else:\n"
    "    print(\"\u26a0 V2.8 cache dataset not mounted \u2014 features will be extracted from scratch\")",
    "else:\n"
    "    print(\"\u26a0 V2.8 cache dataset not mounted \u2014 features will be extracted from scratch\")"
    + _V3_CACHE_BLOCK,
    "add V3 cache preload block",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 9: Recompute ensemble_similarity_cache after calibration write-back
# ═══════════════════════════════════════════════════════════════════════════════
# BUG: Cell 28 computes ensemble_similarity_cache with the initial (hard-coded)
# SPECIES_CONFIG weights before calibration runs. Cell 30 calibrates and updates
# SPECIES_CONFIG, but never recomputes the cache. Cell 33 then clusters using the
# stale pre-calibration ensemble matrix — calibrated weights are completely ignored.
# Only threshold_cluster (also written back) was being applied correctly.
#
# Fix: append a recomputation block at the end of Cell 30, immediately after the
# write-back loop, so the cache reflects the calibrated weights before Cell 33.
# compute_ensemble_similarity_matrix() is pure numpy (fast, <1s per species).
patch_cell(
    nb,
    "print(\"\\n\u2713 Weights + thresholds updated -- Cell 6.2 will use these values\")",
    "print(\"\\n\u2713 Weights + thresholds updated -- Cell 6.2 will use these values\")\n"
    "\n"
    "# Recompute ensemble similarity matrices with calibrated weights.\n"
    "# Cell 28 built ensemble_similarity_cache using pre-calibration weights;\n"
    "# this one-liner refreshes it so Cell 6.2 uses the optimal weights.\n"
    "print(\"\\nRefreshing ensemble_similarity_cache with calibrated weights...\")\n"
    "for _sp in test_meta[\"dataset\"].unique():\n"
    "    ensemble_similarity_cache[_sp] = compute_ensemble_similarity_matrix(_sp)\n"
    "    print(f\"  \u2713 {_sp}\")\n"
    "print(\"\u2713 ensemble_similarity_cache refreshed\")",
    "recompute ensemble cache after calibration",
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
    ("version comment V3.2",
     "V3.2: Fix calib/test score mismatch"),
    ("OLD V3.1 comment gone",
     "V3.1: ALIKED + LightGlue as 3rd local component; V3.0 RANSAC for SIFT/KAZE"),

    # ── Patch 1: compute_local_score_ratio_only ──
    ("ratio_only function defined",
     "def compute_local_score_ratio_only(feat1, feat2"),
    ("ratio_only docstring explains mismatch",
     "Calibration (_calib_match_matrix) uses GPU torch.cdist"),
    ("ratio_only torch.cdist",
     "dists = torch.cdist(desc1, desc2, p=2)"),
    ("ratio_only Lowe ratio test",
     "ratios    = topk_dists[:, 0] / (topk_dists[:, 1] + 1e-8)"),
    ("ratio_only score formula",
     "return float(1.0 - np.exp(-n_matches / 20.0))"),

    # ── Patch 2: dispatch ──
    ("V3.2 dispatch comment",
     "V3.2: LightGlue for ALIKED; ratio-test-only for SIFT/KAZE"),
    ("ratio_only in dispatch",
     "score = compute_local_score_ratio_only(features_list[i], features_list[j])"),
    ("OLD RANSAC dispatch gone",
     "                    score = compute_local_score_gpu(features_list[i], features_list[j])"),

    # ── Patch 3: cache suffix ──
    ("_ratio_matches suffix",
     '"_ratio_matches"'),
    ("OLD _ransac_matches suffix gone",
     '"_ransac_matches"'),

    # ── Patch 4: calibration title ──
    ("calibration title V3.2",
     "V3.2 -- Joint Weight + Threshold Calibration"),
    ("OLD V3.1 calib title gone",
     "V3.1 -- Joint Weight + Threshold Calibration"),

    # ── Patch 5: ALIKEDExtractor seg_mask filter ──
    ("ALIKED extract applies seg_mask",
     "seg_mask = get_seg_mask(str(image_path))"),
    ("ALIKED extract filters kps/descs/scores by mask",
     "kps    = kps[valid]\n                descs  = descs[valid]\n                scores = scores[valid]"),
    ("ALIKED extractor print updated",
     "ALIKED extractor defined (LightGlue, seg_mask filtered)"),

    # ── Patch 6: del _k fix ──
    ("del _k removed",
     "del _sys, _hf_stale\n"),
    ("OLD del _k gone",
     "del _sys, _hf_stale, _k"),

    # ── Patch 7: stale Cell 16 comment ──
    ("Cell 16 updated print",
     "Using SIFT + KAZE + ALIKED (LightGlue) + MiewID v3 ensemble (V3.2)"),
    ("OLD SIFT-only print gone",
     "Using SIFT + MiewID v3 ensemble (robust, compatible approach)"),

    # ── V3.1 structure preserved unchanged ──
    ("lightglue install preserved",
     "git+https://github.com/cvg/LightGlue.git"),
    ("ALIKEDExtractor preserved",
     "from lightglue import ALIKED as _ALIKED_Extractor"),
    ("LightGlue scoring preserved",
     "def compute_local_score_lightglue(feat1, feat2):"),
    ("ALIKED dispatch preserved",
     "score = compute_local_score_lightglue(features_list[i], features_list[j])"),
    ("compute_local_score_gpu preserved (not used for SIFT/KAZE but kept)",
     "def compute_local_score_gpu(feat1, feat2"),
    ("RANSAC code preserved (in compute_local_score_gpu)",
     "cv2.findFundamentalMat"),
    ("_calib_match_matrix preserved",
     "def _calib_match_matrix(descs_list, global_embs, top_k):"),
    ("GPU BFMatcher in calib preserved",
     "torch.cdist(q_d, j_d)"),
    ("_calib_match_matrix_lightglue preserved",
     "def _calib_match_matrix_lightglue(feat_list, global_embs, top_k, device):"),
    ("ALIKED_W_GRID preserved",
     "ALIKED_W_GRID   = [0.0, 0.1, 0.2, 0.3]"),
    ("3D grid search preserved",
     "for aw in ALIKED_W_GRID:"),
    ("kw includes aw preserved",
     "kw = round(1.0 - gw - sw - aw, 4)"),
    ("ensemble includes aliked preserved",
     "gw * global_sim + sw * sift_matrix + kw * kaze_matrix + aw * aliked_matrix"),
    ("calib matrix caching preserved",
     "_gef = Path(f\"{_cp}_global_embs.npy\")"),
    ("CALIB_MAX_IMGS per-species preserved",
     "\"SalamanderID2025\":  9999"),
    ("aliked_weight in calibrated_config preserved",
     '"aliked_weight":  best_aw,'),
    ("aliked write-back preserved",
     'SPECIES_CONFIG[sp]["local_weights"]["aliked"] = cal["aliked_weight"]'),
    ("aw in final print preserved",
     "aw={lw.get('aliked', 0):.2f}"),
    ("V2.8 yellow mask preserved",
     "if 'SalamanderID2025' in rel_key:"),
    ("huggingface_hub fix preserved",
     "huggingface_hub==0.36.2\" --force-reinstall --upgrade"),
    ("sys.modules flush preserved",
     "startswith('huggingface_hub')"),
    ("CALIB_KPT_CAP preserved",
     "CALIB_KPT_CAP   = 300"),
    ("KAZEExtractor preserved",
     "class KAZEExtractor"),
    ("SIFTExtractor preserved",
     "class SIFTExtractor"),
    ("_lg_matches suffix preserved",
     '"_lg_matches"'),
    ("scipy linkage preserved",
     "_sp_linkage(dist_sq[_triu_idx], method=\"average\")"),

    # ── Patch 8: V3 cache preload ──
    ("V3 cache candidates defined",
     "\"/kaggle/input/datasets/svrresearch/animalclef-v3-cache\""),
    ("V3 cache embeddings copy",
     "_e_src = os.path.join(_V3_SRC, \"cache\", \"embeddings\")"),
    ("V3 cache local_features sift/kaze copy",
     "_lf_src = os.path.join(_V3_SRC, \"cache\", \"local_features\")"),
    ("V3 cache skips aliked pkl",
     "if \"_aliked.pkl\" in _fname:"),
    ("V3 cache calib_cache copy",
     "_cc_src = os.path.join(_V3_SRC, \"calib_cache\")"),
    ("V3 cache calib_dst defined",
     "_cc_dst = \"/kaggle/working/calib_cache\""),

    # ── Patch 9: ensemble cache recompute after calibration ──
    ("ensemble cache refresh loop",
     "ensemble_similarity_cache[_sp] = compute_ensemble_similarity_matrix(_sp)"),
    ("ensemble cache refresh print",
     "ensemble_similarity_cache refreshed"),
]

ABSENT = {
    "OLD V3.1 comment gone",
    "OLD RANSAC dispatch gone",
    "OLD _ransac_matches suffix gone",
    "OLD V3.1 calib title gone",
    "OLD del _k gone",
    "OLD SIFT-only print gone",
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
    print("All checks passed \u2014 V3.2 notebook ready for submission.")
