# AnimalCLEF 2026 Solution Comparison: v1 → v2 → v3 → v4 → v5

## Score History (best per version)

| Version | Score | Notes |
|---------|-------|-------|
| V1 | 0.30655 | MiewID + SIFT |
| V1.3 | 0.32528 | SAM3 keypoint mask |
| V2.2 | 0.36607 | RootSIFT |
| V2.3c | 0.37726 | KAZE for all species |
| V2.5 | 0.44747 | KAZE all species + threshold calibration |
| V2.6 | 0.45498 | Joint weight+threshold calibration |
| V2.8 | 0.46202 | Salamander yellow+specular mask |
| V3.2.2 | **0.47912** | **Best overall** — threshold ceiling fix, KAZE-explicit grid |
| V3.2.3 | 0.34934 | THL threshold tuning (catastrophic regression) |
| V4 Run1 | 0.27624 | MiewID + k-RR (threshold ceiling hit) |
| V4 Run2 | 0.15913 | k-RR overfit to val (Lynx under-split on test) |
| V5.11 | 0.38317 | Fine-tuned MegaDesc + Salamander ablation (regression vs V3.2.2) |

## Quick Comparison

| Feature | v1 | v2 | v3 | v4 |
|---------|----|----|-----|-----|
| **Global features** | MiewID v3 | MiewID v3 | MegaDescriptor-L-384 | **MiewID v3 (back)** ✨ |
| **Local features** | SIFT only | SIFT, SP, ALIKED, DISK | SIFT, SP, ALIKED, DISK | **DISK + ALIKED only (Sea Turtle)** |
| **Score calibration** | ❌ None | ❌ None | ✅ Isotonic regression | ❌ **Removed** (collapsed similarities) |
| **Re-ranking** | ❌ None | ❌ None | ❌ None | ✅ **k-Reciprocal Re-ranking** ✨ |
| **Grid search metric** | mAP+P@1+AMI | mAP+P@1+AMI | AMI | **ARI (actual metric)** ✨ |
| **Grid search data** | val | val | val | **train** ✨ |
| **Image size** | 512×512 | 512×512 | 384×384 | **440×440** |
| **Embed dim** | 768 | 768 | 1280 | **2152** |
| **Actual score** | **0.30655** | — | 0.15812 | Run 1: **0.27624** / Run 2: 0.15913 |

---

## Evolution Timeline

### v1: Initial Baseline
**Goal**: Get a working submission with global + local features

**Architecture**:
- Global: MiewID v3 (general animal classification)
- Local: SIFT only
- Ensemble: Simple weighted average (global=0.7, local=0.3)

**Issues**:
- MiewID not optimized for re-ID (classification task)
- Only one local feature extractor (SIFT)
- No score normalization → incomparable similarity scales
- Saved all local features to disk (~4.5 GB)

**Result**: 0.28-0.30 on leaderboard

---

### v2: Multi-Extractor + Disk Optimization
**Goal**: Improve local matching and reduce disk usage

**Key improvements**:
1. **Multi-extractor local features**: Added SuperPoint, ALIKED, DISK
2. **On-the-fly extraction**: Eliminated ~4 GB disk usage
3. **Diverse grid search**: Tested emphasized/de-emphasized weights
4. **Top-K speed optimization**: Only match top-K global candidates

**Architecture**:
- Global: MiewID v3 (same as v1)
- Local: SIFT, SuperPoint, ALIKED, DISK with LightGlue
- Ensemble: Grid-searched weights (global ~65-70%, local ~30-35%)

**Issues still present**:
- MiewID still suboptimal (not purpose-built for re-ID)
- Raw similarity scores not calibrated → scale mismatch
- Missing state-of-the-art techniques from wildlife re-ID literature

**Result**: 0.32-0.36 on leaderboard (+4-6% over v1)

---

### v3: MegaDescriptor + WildFusion Calibration ✨
**Goal**: Apply state-of-the-art wildlife re-ID research

**Key improvements**:
1. **MegaDescriptor-L-384**: Foundation model for wildlife re-ID
   - Trained specifically on 80% of wildlife datasets
   - 75.5% baseline accuracy (vs MiewID's general classification)
   - Purpose-built for individual identity discrimination

2. **WildFusion calibration**: Isotonic regression score normalization
   - Transforms raw similarities → [0,1] probabilities
   - Enables meaningful fusion of heterogeneous scores
   - +8.5% gain in WildFusion paper

3. **Calibrated ensemble**: Grid search on normalized scores
   - Weights become more interpretable
   - Optimal fusion after calibration

**Architecture**:
```
Test Image
    │
    ├─→ MegaDescriptor-L-384 → Cosine Sim → CALIBRATE → [0,1]
    │                                           │
    └─→ Local (SIFT/SP/...)  → LightGlue  → CALIBRATE → [0,1]
                                                │
                                                ▼
                                        Weighted Fusion
                                                │
                                                ▼
                                        Known / Unknown
```

**Expected result**: 0.35-0.40 on leaderboard (+8-13% over v1, +3-7% over v2)

---

## Detailed Comparison

### Global Features

| Aspect | v1 & v2 | v3 |
|--------|---------|-----|
| **Model** | `conservationxlabs/miewid-msv3` | `BVRA/MegaDescriptor-L-384` |
| **Training task** | Animal classification (1000+ species) | Wildlife re-identification |
| **Embedding dim** | 768 | 1280 |
| **Input size** | 512×512 | 384×384 |
| **Baseline accuracy** | ~72% (ImageNet transfer) | 75.5% (wildlife re-ID) |
| **Purpose** | General animal recognition | Individual identity matching |

**Why MegaDescriptor is better**:
- Trained on wildlife re-ID task (not classification)
- Handles identity discrimination (subtle differences)
- Native support for AnimalCLEF2026 dataset

### Score Calibration

| Aspect | v1 & v2 | v3 |
|--------|---------|-----|
| **Method** | None (raw similarities) | Isotonic regression |
| **Global scores** | Uncalibrated cosine [0,1] | Calibrated probability [0,1] |
| **Local scores** | Uncalibrated inlier count | Calibrated probability [0,1] |
| **Ensemble** | Weighted avg of incomparable scores | Weighted avg of calibrated probabilities |

**Example issue without calibration**:
```python
# v2: Incomparable scales
global_sim = 0.85  # cosine similarity
local_sim = 0.30   # inlier-based score
ensemble = 0.7 * 0.85 + 0.3 * 0.30  # ??? What does this mean?

# v3: Calibrated probabilities
global_prob = 0.92  # P(same identity | cosine=0.85)
local_prob = 0.78   # P(same identity | inliers=20)
ensemble = 0.7 * 0.92 + 0.3 * 0.78  # Clear: probability of same identity
```

### Local Features

| Extractor | v1 | v2 & v3 | Description |
|-----------|----|---------| ------------|
| **SIFT** | ✅ | ✅ | CPU-based, robust, multi-threaded |
| **SuperPoint** | ❌ | ✅ | Learned detector/descriptor (GPU) |
| **ALIKED** | ❌ | ✅ | Efficient learned features (GPU) |
| **DISK** | ❌ | ✅ | Dense learned features (GPU) |

All use LightGlue for matching (v2 & v3).

### Ensemble Weights

**v1**: Fixed weights
```python
global = 0.70, local = 0.30  # SIFT only
```

**v2**: Grid-searched, diverse distributions
```python
# Example best config (SeaTurtleID2022)
global = 0.65
local = {
    "sift": 0.15,
    "superpoint": 0.10,
    "aliked": 0.05,
    "disk": 0.05
}
```

**v3**: Grid-searched on **calibrated** scores
```python
# After calibration, weights might shift (more balanced)
global = 0.60  # Lower global weight (calibration equalizes scales)
local = {
    "sift": 0.18,
    "superpoint": 0.12,
    "aliked": 0.06,
    "disk": 0.04
}
```

### Disk Usage

| Component | v1 | v2 | v3 |
|-----------|----|----|-----|
| Global features | 200 MB | 200 MB | 250 MB (MegaDescriptor larger) |
| Local features (saved) | 4 GB | 0 MB | 0 MB |
| Match scores (val) | 50 MB | 50 MB | 50 MB |
| Calibrators | - | - | **1 MB** |
| **Total** | **~4.5 GB** | **~250 MB** | **~300 MB** |

---

## Performance Predictions

### Expected Leaderboard Scores

Based on literature and component analysis:

| Version | Score | Gain | Confidence |
|---------|-------|------|-----------|
| v1 | 0.28-0.30 | Baseline | Actual |
| v2 | 0.32-0.36 | +4-6% | Actual |
| v3 | 0.35-0.40 | +8-13% | **Predicted** |

### Component Contributions (v3)

| Component | Contribution | Source |
|-----------|--------------|--------|
| MegaDescriptor (vs MiewID) | +3-5% | MegaDescriptor paper (75.5% vs ~72%) |
| WildFusion calibration | +5-8% | WildFusion paper (+8.5% from calibration) |
| **Combined** | **+8-13%** | Optimistic estimate (some overlap) |

---

## When to Use Each Version

### Use v1 if:
- You want a quick baseline
- Minimal disk space required (<500 MB)
- Only need SIFT local features
- Testing the pipeline

### Use v2 if:
- You want strong performance without external dependencies
- MiewID is sufficient for your species
- You need multiple local extractors
- Disk space is limited (~250 MB)

### Use v3 if: ✅ **RECOMMENDED**
- You want state-of-the-art wildlife re-ID
- You're willing to install `wildlife-tools`
- You need the best possible accuracy
- You're targeting top leaderboard positions

---

## Migration Path

### v1 → v2
1. Replace Cell 9: Add LightGlue extractors (SuperPoint, ALIKED, DISK)
2. Modify Cell 11: On-the-fly extraction (no disk saves)
3. Update Cell 13: Diverse weight grid search
4. No breaking changes, backward compatible

### v2 → v3
1. Cell 1: Add `pip install wildlife-tools`
2. Cell 2: Add `from sklearn.isotonic import IsotonicRegression`
3. Cell 5-6: Replace MiewIDWrapper with MegaDescriptor
4. Cell 11: Add calibrator fitting
5. Cell 12: Save calibrators
6. Cell 13: Update grid search to use calibrated scores
7. Cell 16: Apply calibration before ensemble

**Breaking changes**:
- Cache files have `_megadesc.npy` suffix (v3) vs no suffix (v2)
- New calibrators directory required

---

---

---

## V1.3 — SAM3 Keypoint Mask Filtering (Best Score)

**Actual score: 0.32528** (+1.9% over V1)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 (768-dim, L2-normalized) |
| Local | SIFT on original image → filter keypoints by SAM3 mask |
| Ensemble | 0.70 × global_sim + 0.30 × BFMatcher score |
| SAM3 coverage | LynxID 0%, Salamander 100%, SeaTurtle 100%, TexasHornedLizard 100% |

Key lesson: filter keypoints by mask *after* SIFT (not before) to avoid boundary artifacts.

---

## V2.0 — MegaDescriptor + RootSIFT + LNBNN (Catastrophic)

**Actual score: 0.10691** (-67% from V1.3 best)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 + MegaDescriptor-L-384 fused → 3688-dim |
| Local | RootSIFT descriptors + LNBNN scoring (replaces BFMatcher) |
| Ensemble | 0.70 × global_sim + 0.30 × LNBNN_score |

**Root cause of failure:** LNBNN with K=100 candidate pool (~100k descriptors) produces near-zero scores for most pairs. Ensemble collapses to `0.70 × global_sim`. V1.3 clustering thresholds become too strict → every image is its own cluster → score ~0.10.

See `FINAL_SOLUTION_v2_0/RESULTS.md` for full analysis.

---

## V2.1 — Diagnostic: RootSIFT + LNBNN only (No MegaDescriptor)

**Actual score: 0.10957** (≈ V2.0 → MegaDescriptor ruled out)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 only (768-dim, unchanged from V1.3) |
| Local | RootSIFT descriptors + LNBNN scoring |
| Purpose | Isolate whether MegaDescriptor or LNBNN caused V2.0 failure |

**Conclusion:** Score nearly identical to V2.0 without MegaDescriptor → LNBNN is the root cause.
MegaDescriptor is not the culprit; it is safe to test it independently.

See `FINAL_SOLUTION_v2_1/RESULTS.md` for full analysis.

---

## V2.2 — RootSIFT + BFMatcher (No LNBNN) ← NEW BEST

**Actual score: 0.36607** (+12.5% over V1.3, new best overall)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 (768-dim, unchanged) |
| Local | RootSIFT descriptors (L1-norm + sqrt) + BFMatcher ratio test (unchanged) |
| Change | One-line transform: `desc /= sum; desc = sqrt(desc)` before mask filter |

**Conclusion:** RootSIFT is a free +4% improvement over raw SIFT with BFMatcher.
Hellinger metric (L2 on sqrt-normalized descriptors) is more discriminative than L2 on raw SIFT.
LNBNN was the sole structural problem in V2.0/V2.1 — not RootSIFT.

See `FINAL_SOLUTION_v2_2/RESULTS.md` for full analysis.

---

## V2.3a — Uniform 0.65/0.35 Weights (Marginal Regression)

**Actual score: 0.36378** (−0.00229 vs V2.2 — slight regression)

| Species | V2.2 | V2.3a |
|---------|------|-------|
| SalamanderID2025 | 0.70/0.30 | 0.65/0.35 |
| SeaTurtleID2022 | 0.65/0.35 | 0.65/0.35 |
| LynxID2025 | 0.70/0.30 | 0.65/0.35 |
| TexasHornedLizards | 0.65/0.35 | 0.65/0.35 |

**Conclusion:** Per-species weight differentiation in V2.2 was already correct.
Salamander (deformable) and Lynx (IR camera trap) benefit from higher global weight.
Do not uniformize. V2.2 weights are the reference going forward.

See `FINAL_SOLUTION_v2_3a/RESULTS.md`.

---

## V2.3c — KAZE for SeaTurtle + TexasHornedLizards ← NEW BEST

**Actual score: 0.37726** (+0.00789 vs V2.3b)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 (unchanged) |
| SeaTurtle local | RootSIFT-SIFT (w=0.20) + KAZE (w=0.15) — unchanged from V2.3b |
| Texas local | RootSIFT-SIFT (w=0.20) + KAZE (w=0.15) — **new** |
| KAZE | `cv2.KAZE_create`, 64-dim float, SAM3 mask, no RootSIFT (signed descriptors) |
| Other species | Salamander and Lynx unchanged (SIFT-only) |

**Key insight:** KAZE gave +2.1% for THL vs only +0.9% for SeaTurtle. Local features carry
more discriminative signal for TexasHornedLizards — its dense spot/scale patterns produce
larger inter-individual variation at the keypoint level. THL is the species most likely
to benefit from further local feature improvements.

See `FINAL_SOLUTION_v2_3c/RESULTS.md`.

---

## V2.3b — KAZE for SeaTurtle

**Actual score: 0.36937** (+0.00330 vs V2.2)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 (unchanged) |
| SeaTurtle local | RootSIFT-SIFT (w=0.20) + KAZE (w=0.15) |
| KAZE | `cv2.KAZE_create`, 64-dim float, SAM3 mask, **no RootSIFT** (signed descriptors) |
| Other species | Unchanged from V2.2 |

**Bug caught:** RootSIFT on KAZE → NaN (signed M-SURF descriptors) → 0 matches silently.
Fixed before submission.

**Conclusion:** KAZE finds complementary keypoints to SIFT on SeaTurtle shell plates.
Next: try KAZE for TexasHornedLizards (also rigid body, 100% SAM3 coverage).

See `FINAL_SOLUTION_v2_3b/RESULTS.md`.

---

## v4: MIEWID v3 + k-RR + ARI Grid Search on Training Data

**Notebook:** `FINAL_SOLUTION_v4/ensemble_miewid_v4.ipynb`

**Why v3 failed (0.15812):**
- Switched to MegaDescriptor — actually weaker than MIEWID for this task (59.2% vs 78.4% top-1)
- Isotonic calibration with <2% positive pairs on val → mapped all similarities to ~0 → all singletons
- Optimized AMI (rewards tight clusters) instead of ARI (the actual competition metric) → pushed threshold to 0.50 → over-splitting

**v4 changes:**
1. **Back to MIEWID v3** — stronger backbone, 2152-dim embeddings, 440×440 input
2. **k-Reciprocal Re-ranking** — enforces mutual nearest-neighbor symmetry before clustering; val ARI 0.08 → 0.80 for SeaTurtle
3. **Grid search on training data** — val has <2% positive pairs; training has real ARI signal
4. **Optimize ARI directly** — not AMI or mAP; ARI is the actual leaderboard metric
5. **Removed isotonic calibration** — MIEWID is already L2-normalized; calibration was the root cause of v3's all-singleton failure

### v4 Run 1 Results (2026-02-18)
**Score: 0.27624** — below V1 (0.30655) due to Lynx over-splitting.

| Species | Train ARI | Val ARI (after k-RR) | Val clusters / images |
|---------|-----------|---------------------|----------------------|
| SeaTurtleID2022 | 0.758 | **0.796** | 262 / 1497 |
| LynxID2025 | 0.046 | 0.050 | **317 / 899** (over-split) |
| SalamanderID2025 | 0.201 | 0.199 | 196 / 272 |

**Root cause of 0.27624:** Threshold search range `[0.10, 0.675]` was too narrow.
Grid search hit the ceiling — Lynx and SeaTurtle both selected threshold=0.675 (max),
and λ=0.40 (max) for all species. Lynx produced 317 clusters for ~15 true val identities
(21× over-split), dragging down the submission score.

### v4 Run 2 Results (2026-02-19) — REGRESSION
**Score: 0.15913** — worse than Run 1 (0.27624), essentially identical to V3's 0.15812.

Extended `threshold_range` to `[0.10, 0.95]` and `kr_lambda_range` to `[0.10 ... 0.60]`.

| Species | Val ARI (after k-RR) | Val clusters / images | Change from Run 1 |
|---------|---------------------|----------------------|-------------------|
| SeaTurtleID2022 | **0.846** | 201 / 1497 | +5% ARI ✓ |
| LynxID2025 | **0.238** | 23 / 899 | +376% ARI but **over-merged on test** |
| SalamanderID2025 | 0.199 | 207 / 272 | unchanged |

**Root cause of regression:** Lynx threshold=0.850 caused massive under-clustering on the test set.

- Val check showed 23 clusters / 899 val images (ratio 1.5× vs ~15 true val ids — looked fine)
- But val identities share training/val pool → many images per identity → dense k-RR neighborhoods → threshold=0.850 safely merges same-id val pairs
- **Test identities are completely unknown individuals** → fewer images per identity → sparser k-RR neighborhoods → Jaccard leaks across different individuals → threshold=0.850 merges everything into ~10-30 giant clusters → ARI collapses

The two failure modes are symmetric:
- Run 1 (0.675): ~330 Lynx test clusters (21× over-split) → ARI ≈ 0.276
- Run 2 (0.850): ~10-30 Lynx test clusters (3-10× under-split) → ARI ≈ 0.159

**True optimal Lynx threshold: ~0.745-0.780** (midpoint between 0.675 and 0.850).
Val sanity check is not reliable for Lynx — val/train identity distribution ≠ test distribution.

**See `FINAL_SOLUTION_v4/EXPERIMENT_LOG.md` for full details.**

### v4 Next Steps (Run 3)
- Hard-override Lynx threshold to ~0.760 (not grid-searched — val signal unreliable for Lynx)
- Optionally add cluster-count floor constraint: reject any combo that gives `n_clusters > C * n_val_images` for Lynx

---

---

## V2.5 — KAZE All Species + Threshold Calibration ← New Best (+18.7% vs V2.3c)

**Actual score: 0.44747** (+0.07021 vs V2.3c)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 (unchanged) |
| Local | SIFT + KAZE for ALL 4 species |
| Calibration | Per-species AMI grid-search over threshold values |
| Lynx KAZE | Original images (0% SAM3) — no mask filter |

**Why such a large jump:** Threshold calibration per species (rather than fixed thresholds) was the main driver. KAZE for Lynx/Salamander added incremental signal on top.

---

## V2.6 — Joint Weight+Threshold Calibration ← New Best

**Actual score: 0.45498** (+0.00751 vs V2.5)

| Feature | Detail |
|---------|--------|
| Calibration | Grid-search over (gw, sw, kw, threshold) jointly |
| Grid | GLOBAL_W_GRID × SIFT_W_GRID × KAZE_W_GRID × THR_GRID |
| Metric | AMI (reasonable proxy for ARI) |

**Calibrated weights per species:**
- Lynx: gw=0.75, sw=0.10, kw=0.15, thr=0.59 (local features noisy for IR camera trap)
- Salamander: gw=0.65, sw=0.25, kw=0.10, thr=0.39
- SeaTurtle: gw=0.75, sw=0.10, kw=0.15, thr=0.51
- THL: gw=0.75, sw=0.10, kw=0.15, thr=0.30 (zero-shot, uncalibrated)

---

## V2.7 — XFeat Replacing KAZE (Regression)

**Actual score: 0.37257** (−0.08241 vs V2.6)

XFeat (learned extractor) replaced KAZE for all species. XFeat was trained on human/urban scenes — its descriptors don't generalise to biological textures. KAZE wins on animal skin/scale patterns.

**Lesson: domain alignment matters more than "learned vs classical."**

---

## V2.8 — Salamander Yellow+Specular Mask ← New Best

**Actual score: 0.46202** (+0.00704 vs V2.6)

| Feature | Detail |
|---------|--------|
| Change | `get_seg_mask()` adds HSV yellow+specular filter for Salamander only |
| Specular | V>220 & S<40 |
| Yellow | H∈[18,42] & S>80 & V>80 |
| Dilation | 25px |
| Bug fix | `_extract_calib_local` was passing relative path to `get_seg_mask()` → always None |
| CALIB_KPT_CAP | 300 caps keypoints during calibration only (8× speedup) |

**Lesson:** Species-specific mask post-processing (removing false colour patches) helps SIFT/KAZE match quality. Calibration must use the same mask logic as inference.

---

## V3.1 — ALIKED + LightGlue as 3rd Local Component (Regression)

**Actual score: ~0.41500** (−0.04702 vs V2.8)

| Feature | Detail |
|---------|--------|
| Addition | ALIKED (128-dim learned) + LightGlue matcher |
| SIFT/KAZE matching | Ratio test + RANSAC geometric gate |
| Grid | GLOBAL_W_GRID × SIFT_W_GRID × ALIKED_W_GRID (kw=1-gw-sw-aw residual) |

**Root cause of regression:** SIFT/KAZE calibration used ratio-only scores (same as V2.8), but test pipeline applied RANSAC gate → score distribution mismatch → calibration assigned kw=0 for all species → KAZE disabled at test time → net regression vs V2.8.

---

## V3.2 — Ratio-Only SIFT/KAZE + Bug Fixes

**Actual score: 0.42134** (below V2.8 still)

| Feature | Detail |
|---------|--------|
| SIFT/KAZE | `compute_local_score_ratio_only()` — consistent at calib AND test time |
| ALIKED | seg_mask passed to extract() for keypoint filtering |
| Bug fix | `del _k` NameError in ALIKED extraction loop |
| Patch 9 | Recompute ensemble_similarity_cache AFTER calibration write-back (was using stale pre-calib weights for clustering) |
| Cache suffix | `_ratio_matches` (not `_ransac_matches`) |
| Grid | KAZE still crowded to 0 by ALIKED-explicit / KAZE-residual grid structure |

**Still below V2.8 because:** Grid had ALIKED explicit, KAZE as residual → calibration crowds KAZE to 0.00. Threshold ceiling still at 0.59 (Lynx optimum = 0.65).

---

## V3.2.2 — Threshold Ceiling Fix + KAZE-Explicit Grid ← **NEW BEST**

**Actual score: 0.47912** (+0.01710 vs V2.8, **best overall**)

| Feature | Detail |
|---------|--------|
| Threshold grid | Extended to [0.15..0.75] with 31 steps |
| Grid structure | KAZE explicit, ALIKED residual: `aw=1-gw-sw-kw` |
| THL fix | Restored V2.8 weights (gw=0.75, sw=0.10, kw=0.15, aw=0.00, thr=0.30) |

### Best Calibrated Weights

| Species | gw | sw | kw | aw | thr | clusters/N |
|---------|----|----|----|----|-----|-----------|
| LynxID2025 | 0.40 | 0.00 | 0.05 | **0.55** | **0.65** | 38/946 |
| SalamanderID2025 | 0.50 | 0.30 | 0.15 | 0.05 | 0.47 | 277/689 |
| SeaTurtleID2022 | 0.70 | 0.20 | 0.05 | 0.05 | 0.57 | 119/500 |
| TexasHornedLizards | 0.55 | 0.15 | 0.10 | 0.20 | 0.30 | 155/274 |

**Key insight:** Lynx ALIKED weight (aw=0.55) is the highest of any component for any species. ALIKED finds meaningful texture patches for Lynx even without segmentation, while SIFT/KAZE mostly detect background noise.

### Score Distribution Stats (off-diagonal, test set)

| Species | SIFT mean/std | KAZE mean/std | ALIKED mean/std |
|---------|---------------|---------------|-----------------|
| Lynx | 0.030/0.122 | 0.045/0.178 | 0.049/0.187 |
| Salamander | 0.025/0.091 | 0.036/0.121 | 0.056/0.186 |
| SeaTurtle | 0.018/0.071 | 0.013/0.063 | 0.067/0.209 |
| THL | 0.055/0.117 | 0.053/0.119 | **0.225/0.390** |

THL ALIKED anomalously high mean (0.22) but also very high std (0.39) = discriminative, not noise.

---

## V3.2.3 — THL Threshold Tuning (Catastrophic Regression)

**Actual score: 0.34934** (−0.12978 vs V3.2.2)

Changed THL thr=0.30→0.40 based on stability sweep analysis → produced 57 clusters.
Ground truth THL is ~150-200 individuals → massive over-merging → score collapsed.

**Critical lesson:** THL ALIKED (std=0.39) IS discriminative. DO NOT remove or reduce aw for THL.
Keep thr=0.30, aw=0.20 → ~155 clusters for 274 images. Never tune zero-shot thresholds
based on stability sweeps alone without validating cluster count vs ground truth estimates.

---

---

## V5.1 — Fine-tuned MegaDescriptor-T-224 + Dual Global + AMI Calibration

**Actual score: unknown** (baseline for V5.11 comparison)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 (2152-dim) + fine-tuned MegaDescriptor-T-224 per species |
| Local | SIFT + KAZE + ALIKED (all species) |
| Fine-tuning | ArcFace on species train splits — Lynx, Salamander, SeaTurtle |
| Calibration | Joint (mw, mgw, sw, kw, aw, thr) grid-search on training data, AMI metric |
| Default Salamander | mw=0.40 mgw=0.00 sw=0.40 kw=0.15 aw=0.05 thr=0.51 (AMI=0.42) |

**Key change from V3.2.2:** Introduced second global model (fine-tuned MegaDesc). Lynx default switched to ALIKED-dominant (aw=0.55), which was the V3.2.2 finding.

---

## V5.11 — Salamander KAZE kpt Ablation + Bug Fixes

**Actual score: 0.38317** (−0.09595 vs V3.2.2 best; regression)

| Feature | Detail |
|---------|--------|
| Global | MiewID v3 + fine-tuned MegaDescriptor-T-224 (same as V5.1) |
| Local | SIFT + KAZE + ALIKED — Salamander: SIFT nf=500, KAZE nk=2000 (ablation-optimal) |
| Bug fixes | `sp` NameError in `_extract_calib_local/aliked`; `from pathlib import Path`; duplicate `import timm` |
| Skip MegaDesc | Species with `mega_weight=0.00` skip extraction entirely (saves compute) |

### Calibrated Weights (from run output)

| Species | mw | mgw | sw | kw | aw | thr | clusters/N | Train AMI |
|---------|----|----|----|----|-----|-----|-----------|-----------|
| LynxID2025 | 0.10 | **0.50** | 0.05 | 0.00 | 0.35 | 0.57 | — | 0.6898 |
| SalamanderID2025 | 0.40 | 0.00 | 0.40 | 0.15 | 0.05 | 0.51 | 289/689 | 0.4212 |
| SeaTurtleID2022 | 0.50 | 0.20 | 0.15 | 0.10 | 0.05 | 0.63 | 94/500 | 0.8584 |
| TexasHornedLizards | 0.28 | 0.28 | 0.15 | 0.10 | 0.20 | 0.30 | 201/274 | — |

### Root Cause of Regression vs V3.2.2

1. **Salamander KAZE ablation didn't take effect**: KAZE features loaded from V5.9 cache (old nk=300 or 1000). Calibration found kw=0.15 is optimal with old features — reverted to V5.1 weights. The nk=2000 improvement only applies if features are recomputed fresh.

2. **Lynx dominated by fine-tuned MegaDesc (mgw=0.50)**: V3.2.2's key insight was Lynx ALIKED (aw=0.55) is the most discriminative component. V5.11 calibration shifted to mgw=0.50 (fine-tuned MegaDesc) — possibly overfitting to Lynx training identities, hurts on test unknowns.

3. **Lynx threshold dropped 0.67 → 0.57**: Tighter threshold → more clusters. With mgw=0.50 (potentially overtrained), this likely over-splits Lynx test set.

### Lessons Learned

- **Ablation cache isolation**: If testing new kpt counts (nk=2000), must delete old Salamander KAZE cache so features are recomputed. Otherwise calibration finds old-feature-optimal weights.
- **Fine-tuned MegaDesc for Lynx is risky**: Camera trap IR images with few training examples → ArcFace likely memorises training IDs → weak generalisation to test unknowns. V3.2.2 showed pure MiewID + ALIKED (no MegaDesc) at 0.47912.
- **Lynx mgw=0.50 needs ablation**: Test with Lynx mgw forced to 0.00 (V3.2.2 baseline) to isolate impact.

### Next Steps (V5.12)

- Delete Salamander SIFT/KAZE cache before submitting so nf=500/nk=2000 features are actually used
- Try Lynx with mgw=0.00 forced (no fine-tuned MegaDesc) → revert to V3.2.2 Lynx config
- Investigate THL cluster count: 201/274 = 73% singleton rate — possibly too many clusters

---

## Future Improvements (Beyond v3)

If v3 still underperforms:

### Priority 1: Foreground/Background Modeling
- Use SAM to separate animal from background
- Per-Instance Temperature Scaling (PITS)
- Requires GPS/timestamp metadata
- **Expected gain**: +10-15% for animals in new locations

### Priority 2: Domain-Specific Features
- TexasHornedLizards: Focus on ventral spot patterns
- Crop belly region → extract SIFT
- **Expected gain**: +2-3% for TexasHornedLizards only

### Priority 3: Ensemble Multiple Global Models
- MegaDescriptor L-384 + B-224
- Late fusion of embeddings
- **Expected gain**: +1-2%

### Priority 4: Re-ranking
- Spatiotemporal filtering (if metadata available)
- Reciprocal nearest neighbors
- **Expected gain**: +2-3%

---

## Key Takeaways

1. **v1 → v2**: Multi-extractor local features (+4-6%)
2. **v2 → v3**: Purpose-built re-ID model + calibration (+3-7%)
3. **Critical insight**: Calibration matters! Raw scores are not comparable.
4. **MegaDescriptor advantage**: Trained for re-ID, not classification
5. **WildFusion insight**: Score normalization enables optimal fusion

**Bottom line**: v3 implements state-of-the-art wildlife re-ID research. Expected to reach 0.35-0.40 on leaderboard.

---

## Files Generated

```
AnimalCLEF_26/
├── FINAL_SOLUTION_v1/          # (if exists)
│   └── ...
├── FINAL_SOLUTION_v2/
│   ├── build_notebook.py
│   └── ensemble_global_local_reid_v2.ipynb
├── FINAL_SOLUTION_v3/          # ✨ NEW
│   ├── build_notebook.py
│   ├── ensemble_wildlife_tools_v3.ipynb
│   └── README.md
└── SOLUTION_COMPARISON.md      # This file
```

---

## Recommendation

**Use FINAL_SOLUTION_v3** for best results. It implements:
- ✅ MegaDescriptor (state-of-the-art wildlife re-ID)
- ✅ WildFusion calibration (proven +8.5% gain)
- ✅ Multi-extractor local features
- ✅ Optimized disk usage (~300 MB)
- ✅ Grid-searched hyperparameters

Expected leaderboard score: **0.35-0.40** (vs 0.32-0.36 baseline)
