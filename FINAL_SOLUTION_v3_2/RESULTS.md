# V3.2 Results

## Overview

V3.2 is the ALIKED+LightGlue + ratio-only SIFT/KAZE version. It went through three live
experiments on the same Kaggle environment.

| Submission | Score | Notes |
|-----------|-------|-------|
| V3.2 (first submit) | 0.42134 | Grid structure error — KAZE crowded to 0 |
| V3.2.2 (live fix) | **0.47912** | **NEW BEST** — threshold ceiling + KAZE-explicit grid |
| V3.2.3 (live fix) | 0.34934 | THL thr=0.40 → 57 clusters (catastrophic regression) |

---

## V3.2 First Submit — 0.42134

### Root Causes
1. **KAZE crowded to 0**: Grid had ALIKED explicit (aw ∈ [0,0.05,0.10,0.15,0.20]) and
   KAZE as residual (kw = 1−gw−sw−aw). Calibration consistently set kw=0.00 even though
   KAZE helps — the grid structure forced it to compete with ALIKED for the residual slot.
2. **Threshold ceiling at 0.59**: Lynx optimum was 0.65 → was over-merging (20 clusters
   instead of ~38).

### Calibration Output (before fix)
- Lynx: kw=0.00 for all calibrated configs
- All species: kw=0.00

---

## V3.2.2 — 0.47912 (New Best)

### Changes Applied Live
1. Extended threshold grid: `np.linspace(0.15, 0.75, 31)` (was `np.linspace(0.15, 0.59, 23)`)
2. Flipped grid: KAZE explicit `KAZE_W_GRID=[0, 0.05, 0.10, 0.15]`, ALIKED residual
   `aw = 1 - gw - sw - kw`
3. THL restored to V2.8 baseline: `gw=0.75, sw=0.10, kw=0.15, aw=0.00, thr=0.30`

### Calibrated Weights

| Species | gw | sw | kw | aw | thr | clusters/N |
|---------|----|----|----|----|-----|-----------|
| LynxID2025 | 0.40 | 0.00 | 0.05 | 0.55 | 0.65 | 38/946 |
| SalamanderID2025 | 0.50 | 0.30 | 0.15 | 0.05 | 0.47 | 277/689 |
| SeaTurtleID2022 | 0.70 | 0.20 | 0.05 | 0.05 | 0.57 | 119/500 |
| TexasHornedLizards | 0.55 | 0.15 | 0.10 | 0.20 | 0.30 | 155/274 |

### Key Observations
- **Lynx aw=0.55**: Highest ALIKED weight of any species. ALIKED finds texture patches
  even without segmentation; SIFT/KAZE detect background noise (hence sw=0.00, kw=0.05).
- **Salamander hitting ceilings**: sw=0.30 is the SIFT_W_GRID ceiling; kw=0.15 is the
  KAZE_W_GRID ceiling. Both are at their maximum allowed values. True optimum likely higher.
- **THL ALIKED anomaly**: mean=0.2249 (vs 0.03-0.07 for other species). High std=0.3898
  means discriminative, not noise. aw=0.20 makes sense.

### Score Distribution Stats (off-diagonal, test set)

| Species | SIFT | KAZE | ALIKED |
|---------|------|------|--------|
| Lynx | mean=0.030, std=0.122 | mean=0.045, std=0.178 | mean=0.049, std=0.187 |
| Salamander | mean=0.025, std=0.091 | mean=0.036, std=0.121 | mean=0.056, std=0.186 |
| SeaTurtle | mean=0.018, std=0.071 | mean=0.013, std=0.063 | mean=0.067, std=0.209 |
| THL | mean=0.055, std=0.117 | mean=0.053, std=0.119 | **mean=0.225, std=0.390** |

---

## V3.2.3 — 0.34934 (Catastrophic Regression)

### What Changed
- Calibration grids extended (SIFT_W_GRID to 0.35, KAZE_W_GRID to 0.20)
  → all species returned SAME weights as V3.2.2 (already not at true ceiling)
- THL threshold overridden: thr=0.30 → 0.40 based on stability sweep "elbow" analysis

### Root Cause
THL thr=0.40 → 57 clusters. THL has 274 test images and ~150-200 true individuals.
57 clusters means massive over-merging → near-random assignment → THL ARI collapses.

### The Stability Sweep Mistake
Stability sweep tested: for various (gw, sw, kw) configs, cluster count at thresholds
0.25 to 0.55. V2.8 weights (gw=0.75) showed steep cluster count decline at thr=0.35-0.40.
This was interpreted as "steep drop = good discrimination → thr=0.40 is right."

**The flaw**: A steep drop tells you there IS separation but not WHERE to put the threshold
relative to the true number of individuals. For zero-shot species (THL), you must use the
estimated ground truth count (~150-200) to anchor the threshold.

V3.2.2 THL config (aw=0.20, thr=0.30 → ~155 clusters) is already well-anchored.

### Lesson
**For zero-shot species: never tune threshold using discriminability analysis alone.
Always cross-check that the resulting cluster count is plausible given image count.**
THL: 274 images / ~1.5-2 imgs per individual → ~140-180 clusters. thr=0.30 gives ~155 ✓.

---

## Bugs Fixed in V3.2 (vs V3.1)

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | SIFT/KAZE ratio-only inconsistency | Calibration score ≠ test score → kw=0 | `compute_local_score_ratio_only()` at both calib and test |
| 2 | ALIKED seg_mask not passed to extract() | All keypoints used (including background) | Pass `seg_mask` to `extract_aliked_features()` |
| 3 | `del _k` NameError | Crash during ALIKED extraction | Fixed variable name |
| 4 | Ensemble cache stale after calibration (Patch 9) | Clustering used pre-calibration weights | Recompute cache at end of Cell 30 |

---

## Next Steps (V3.3)

1. Use V3.2.2 calibrated weights as SPECIES_CONFIG defaults
2. Expand Salamander grid: SIFT_W_GRID up to 0.40, KAZE_W_GRID up to 0.20
3. Add SuperPoint + LightGlue as 4th local component
4. Powell optimizer for continuous weight calibration (fix: use `_fc` not `fcluster`)
5. Consider caching SuperPoint features as `animalclef-v3-2-cache` dataset

## Files

- Build script: `FINAL_SOLUTION_v3_2/build_notebook_v3_2.py`
- Notebook: `FINAL_SOLUTION_v3_2/ensemble_global_local_reid_v3_2.ipynb`
- Kernel metadata: `FINAL_SOLUTION_v3_2/kernel-metadata.json`
- Dataset: `svrresearch/animalclef-v3-cache` (ALIKED pkl + updated calib cache)
