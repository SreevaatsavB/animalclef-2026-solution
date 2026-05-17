# AnimalCLEF 2026 — Complete Experiment Log

All submissions, approaches, outcomes, and lessons. Two parallel tracks ran across the competition:
- **Track A (V1 → V2.x → V3.1+ → V4.0 → V5.x)**: Incremental improvements on MiewID v3 + BFMatcher base, then MegaDescriptor addition and fine-tuning → **current best (V5.1: 0.51705)**
- **Track B (V3 → V7)**: Early ambitious architectural changes (new backbones, ArcFace projection heads, re-ranking) → all regressed or unconfirmed

---

## Score Overview

| Version | Score | Δ from best at time | Status |
|---------|-------|---------------------|--------|
| V1 | 0.30655 | — (baseline) | Stable baseline |
| V1.2 | 0.26330 | −0.04325 | Retired |
| V1.3 | 0.32528 | +0.01873 | Best at time |
| V3(B) | 0.15812 | −0.16716 | Failed (wrong backbone + metric) |
| V2.0 | 0.10691 | −0.21837 | Failed (LNBNN collapse) |
| V2.1 | 0.10957 | −0.21571 | Diagnostic only |
| V4(B) Run 1 | 0.27624 | −0.04904 | Failed (threshold ceiling) |
| V4(B) Run 2 | 0.15913 | −0.16615 | Failed (Lynx over-merge) |
| V2.2 | 0.36607 | +0.04079 | Best at time |
| V2.3a | 0.36378 | −0.00229 | Minor regression |
| V2.3b | 0.36937 | +0.00330 | Best at time |
| V2.3c | 0.37726 | +0.00789 | Best at time |
| V2.4 | 0.35624 | −0.02102 | Regression (AQE) |
| V2.5 | 0.44747 | +0.07021 | Best at time |
| V2.6 | 0.45498 | +0.00751 | Superseded |
| V2.7 | 0.37257 | −0.08241 | Regression (XFeat weights mismatch) |
| V2.8 | 0.46202 | +0.00704 | Superseded |
| V5(B) | ~0.27 (est.) | — | Not confirmed on LB |
| V6(B) | ~0.27 (est.) | — | Not confirmed on LB |
| V7(B) | unknown | — | Not submitted |
| V3.1 | 0.41500 | −0.04702 | Regression (calib/test mismatch) |
| V3.2 | 0.42134 | −0.04068 | Regression (grid error) |
| V3.2.2 | 0.47912 | +0.01710 | Best at time |
| V3.2.3 | 0.34934 | −0.12978 | Catastrophic (THL over-merge) |
| V3.3 | 0.47183 | −0.00729 | Regression (SuperPoint) |
| V4.0-ARI | 0.41769 | −0.06143 | Failed (ARI over-merged Lynx) |
| V4.0 | 0.48600 | +0.00688 | Best at time |
| V5.0 | 0.34825 | −0.13775 | Regression (isotonic + 2-phase) |
| **V5.1** | **0.51705** | **+0.03105** | **Current best** |
| V5.2 | 0.41964 | −0.09741 | Regression (under-trained L-384) |
| V5.4 | 0.47917 | −0.03788 | Regression (calib over-fit) |
| V5.5 | 0.50506 | −0.01199 | Hotfix (still below V5.1) |
| V5.6 | 0.21434 | −0.30271 | Catastrophic (HDBSCAN all-singletons) |
| V5.7 | 0.49050 | −0.02655 | Regression (k-Reciprocal Re-ranking) |
| V5.9 | 0.49132 | −0.02573 | Regression (fine-tuned MiewID+MegaDesc+TTA; memorisation) |
| V5.9b | 0.48379 | −0.03326 | Regression (same, no TTA — still below V5.1) |

---

## Track A — Incremental Improvements

### V1 — Baseline (Score: 0.30655)

**Approach:** MiewID v3 global embeddings + SIFT local features + BFMatcher + AgglomerativeClustering.

**Architecture:**
- Global: `conservationxlabs/miewid-msv3` (EfficientNetV2-RW-M, 2152-dim, L2-normalized)
- TTA: original + horizontal flip → sum → L2 normalize
- Local: `cv2.SIFT_create(nfeatures=500)` raw descriptors
- Matching: `cv2.BFMatcher(cv2.NORM_L2)` + Lowe's ratio test (0.75)
- Score: `1 - exp(-num_matches / 20.0)`
- Ensemble: `global_weight * cosine_sim + local_weight * match_score`
- Clustering: `AgglomerativeClustering(metric="precomputed", linkage="average")`

**Species weights:**
| Species | Global | SIFT |
|---------|--------|------|
| SalamanderID2025 | 0.70 | 0.30 |
| SeaTurtleID2022 | 0.65 | 0.35 |
| LynxID2025 | 0.70 | 0.30 |
| TexasHornedLizards | 0.65 | 0.35 |

**GPU acceleration:** Top-K=100 candidate selection using `torch.cdist` before BFMatcher to avoid O(N²) full matching. Batch size 50.

**Key files:** `FINAL_SOLUTION_v1/ensemble_global_local_reid.ipynb`

---

### V1.2 — SAM3 White-Background (Score: 0.26330, −14%)

**Change:** Before SIFT, replace background pixels with white (255,255,255) using SAM3 segmentation masks.

**Why it failed:** SIFT detected strong edges at the animal/white boundary. These boundary-artifact keypoints are pose-dependent and not identity-discriminative → more false matches between different animals.

**SAM3 coverage:** Salamander 100%, SeaTurtle 100%, THL 100%, Lynx 0%.

**Lesson:** Never replace background before keypoint extraction. Boundary = noise.

**Key files:** `FINAL_SOLUTION_v1_2/ensemble_global_local_reid_v1_2.ipynb`

---

### V1.3 — SAM3 Keypoint Mask Filtering (Score: 0.32528, +6.1%)

**Change (one line):** Run SIFT on original image, then discard keypoints where `seg_mask[y, x] == 0`. Only animal-region keypoints forwarded to matching.

```python
# Correct approach
kpts, desc = sift.detectAndCompute(original_image, None)
kpts = [k for k in kpts if seg_mask[int(k.pt[1]), int(k.pt[0])] > 0]
```

**Why it worked:** Removes background keypoints (trees, rocks, substrate) without disturbing the SIFT computation itself. Descriptors computed on original pixel values → no boundary artifacts.

**Gain source:** Salamander (+100% SAM3), SeaTurtle (+100%), THL (+100%). Lynx unchanged (0% coverage).

**Key files:** `FINAL_SOLUTION_v1_3/ensemble_global_local_reid_v1_3.ipynb`, `FINAL_SOLUTION_v1_3/RESULTS.md`

---

### V2.0 — MegaDescriptor + RootSIFT + LNBNN (Score: 0.10691, −67%)

**Changes (all at once):**
1. Fused global: MiewID v3 (2152-d) + MegaDescriptor-L-384 (1536-d) → 3688-dim concat + L2-normalize
2. RootSIFT: L1-normalize descriptors → element-wise sqrt before BFMatcher
3. LNBNN scoring: HotSpotter-style Local Naive Bayes Nearest Neighbor replacing BFMatcher ratio test

**Bugs caught and fixed before submission (not the cause of failure):**
- `num_classes=0` missing in `timm.create_model` → returned class logits not embeddings
- Self-image in LNBNN `db_descs_list` → background distance distorted
- `np.add.at` accumulation → replaced with GPU `scatter_add_`
- `np.argsort` for top-K → `np.argpartition` (faster)
- Python loop → vectorised `np.maximum` broadcasting

**Root cause (LNBNN):**
```
K=100 candidates × ~1000 keypoints = ~100,000 descriptors in pool
Background distance (4th nearest in 100k similar-animal descriptors) ≈ 0.10–0.15 Hellinger
For most query descriptors: bg_dist < match_dist → delta ≤ 0 → LNBNN score = 0
match_matrix ≈ identity → ensemble = 0.70 × global_sim + 0 → thresholds miscalibrated
Same-individual distance = 1 − 0.70×0.80 = 0.44 > threshold 0.35 → NOT clustered → singletons
```

**Key files:** `FINAL_SOLUTION_v2_0/RESULTS.md`, `FINAL_SOLUTION_v2_0/v2_postmortem.md`

---

### V2.1 — Diagnostic: RootSIFT + LNBNN, no MegaDescriptor (Score: 0.10957)

**Purpose:** Isolate whether MegaDescriptor or LNBNN caused V2.0 failure.

**Result:** Score 0.10957 ≈ V2.0's 0.10691 **without** MegaDescriptor → MegaDescriptor was NOT the culprit. LNBNN confirmed as the sole structural failure.

**Key files:** `FINAL_SOLUTION_v2_1/RESULTS.md`

---

### V2.2 — RootSIFT + BFMatcher (Score: 0.36607, +12.5% over V1.3)

**Change (one line):** Apply RootSIFT transform to SIFT descriptors before BFMatcher. No LNBNN.

```python
desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)  # L1 normalize
desc = np.sqrt(desc)                                      # element-wise sqrt
```

**Why it works:** Converts L2 distance to Hellinger distance, which is more discriminative for gradient-histogram (HOG-style) descriptors. The Hellinger kernel is known to outperform RBF/cosine for histogram features.

**Everything else:** Identical to V1.3 (same model, same BFMatcher, same thresholds, same weights).

**Key insight:** RootSIFT is a free improvement — zero extra compute, zero dependencies, zero risk. The only change is two lines of numpy applied to descriptors at extraction time.

**Key files:** `FINAL_SOLUTION_v2_2/RESULTS.md`

---

### V2.3a — Uniform 0.65/0.35 Weights (Score: 0.36378, −0.6%)

**Change:** Uniformized Salamander and Lynx from 0.70/0.30 to 0.65/0.35 (matching SeaTurtle/THL).

**Result:** Slight regression. Per-species weight differentiation was already better calibrated:
- Salamander: deformable body, subtle local texture → global embedding should dominate (0.70)
- Lynx: 0% SAM3, camera-trap IR images, noisy SIFT → global should dominate (0.70)

**Lesson:** Do not uniformize weights across species. Each species has a different signal-to-noise ratio for local features.

**Key files:** `FINAL_SOLUTION_v2_3a/RESULTS.md`

---

### V2.3b — KAZE for SeaTurtle (Score: 0.36937, +0.9%)

**Change:** Added `KAZEExtractor` for SeaTurtleID2022. New weights: `sift:0.20 + kaze:0.15 + global:0.65`.

**KAZEExtractor design:**
```python
cv2.KAZE_create(upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4)
```
- Non-linear (anisotropic diffusion) scale space — finds different stable keypoints than SIFT's Gaussian pyramid
- 64-dim signed M-SURF descriptors
- **RootSIFT NOT applied** — signed values → `sqrt(negative) = NaN` → silent zero-match failure
- SAM3 mask filter applied (SeaTurtle: 100% coverage)

**Bug caught before submission:** Initial implementation applied RootSIFT to KAZE → NaN descriptors → 0 good matches → no improvement. Fixed by removing RootSIFT from KAZEExtractor path.

**Key files:** `FINAL_SOLUTION_v2_3b/RESULTS.md`

---

### V2.3c — KAZE for TexasHornedLizards (Score: 0.37726, +2.1%)

**Change:** Added KAZE to TexasHornedLizards. New weights: `sift:0.20 + kaze:0.15 + global:0.65`.

No new code — KAZEExtractor already existed from V2.3b. Only `SPECIES_CONFIG` patched.

**Why THL benefits more than SeaTurtle (+2.1% vs +0.9%):**
- THL has extremely dense spot/scale patterns with high-frequency texture
- KAZE's non-linear scale space finds stable keypoints where SIFT's Gaussian pyramid blurs them
- Each individual has a unique spot arrangement → high inter-individual variation in local texture
- 100% SAM3 coverage means every image benefits (no background keypoints)

**Key files:** `FINAL_SOLUTION_v2_3c/RESULTS.md`, `FINAL_SOLUTION_v2_3c/build_notebook_v2_3c.py`

---

### V2.4 — Alpha Query Expansion on Global Embeddings (Score: 0.35624, −5.6%)

**Approach:** Post-processing on cached V2.3c global embeddings. No new feature extraction.

**Formula:**
```
e_i' = L2_normalize(e_i + Σ_{j ∈ top-k(i)} sim(i,j)^α · e_j)
```

**Per-species params** (chosen by within-vs-between separation metric on V2.3c pseudo-labels):

| Species | k | α | Separation improvement |
|---------|---|---|----------------------|
| SalamanderID2025 | 2 | 3.0 | +4.1% |
| SeaTurtleID2022 | 2 | 3.0 | +6.7% |
| LynxID2025 | 2 | 5.0 | +2.5% |
| TexasHornedLizards | 1 | 3.0 | +2.4% |

**Why it failed (-5.6%):**
- MiewID v3 embeddings are already near-optimal for this task — AQE over-smooths them
- Salamander: −105 clusters (over-merging — AQE pulled different-individual embeddings together)
- Lynx: +71 clusters (over-splitting — AQE disturbed tight within-individual clusters)
- The "separation metric" improvement was a false signal (V2.3c pseudo-labels as proxy GT)
- Agreement with V2.3c: 99.2–100% of pairs — real changes but wrong direction

**Key lesson:** Do not apply post-processing to global embeddings for this dataset. The embedding space is already optimal; perturbation only hurts.

**Key files:** `FINAL_SOLUTION_v2_4/run_v2_4.py`, `FINAL_SOLUTION_v2_4/submission_v2.4.csv`

---

### V2.5 — KAZE All Species + Threshold Calibration (Score: 0.44747, +18.6%)

**Changes from V2.3c:**
1. KAZE added to SalamanderID2025: `global:0.65 + sift:0.20 + kaze:0.15`
2. KAZE added to LynxID2025: `global:0.70 + sift:0.20 + kaze:0.10` (conservative — 0% SAM3)
3. All local features preloaded from `sreevaatsavbavana/v2-5-cache` Kaggle dataset (16 files)
4. **Threshold calibration** via AMI grid-search on training identity labels

**Threshold calibration method:**
- Extract MiewID v3 global embeddings for training images (GPU, TTA)
- Compute cosine similarity on training set
- Grid-search: threshold ∈ [0.15, 0.60], step 0.01 → 46 values
- Select threshold maximising `adjusted_mutual_info_score(true_labels, pred_clusters)`
- THL: no training split → keep V2.3c threshold

**Calibration results:**

| Species | Train images | Identities | Old threshold | New threshold | Δ | AMI | Time |
|---------|-------------|-----------|---------------|---------------|---|-----|------|
| LynxID2025 | 2,500 / 2,957 | 76 | 0.35 | **0.51** | +0.16 | 0.4269 | 163.8s |
| SalamanderID2025 | 1,388 | 587 | 0.35 | **0.29** | −0.06 | 0.3058 | 104.8s |
| SeaTurtleID2022 | 2,500 / 8,729 | 412 | 0.40 | **0.42** | +0.02 | 0.8397 | 118.6s |
| TexasHornedLizards | — | — | 0.30 | 0.30 | 0 | — | — |

**Why Lynx gained most (+0.16 threshold shift):** The old threshold of 0.35 required cosine similarity ≥ 0.65 to merge two images. Within-individual Lynx pairs (varying camera angles, seasons) typically score 0.49–0.65 — these were NOT being merged, producing ~300+ over-split clusters. Raising to 0.51 (merges similarity ≥ 0.49) correctly consolidates them into 53 clusters.

**Why Salamander threshold went DOWN (0.35 → 0.29):** Different-individual Salamander pairs have similarity close to 0.65–0.71 (deformable bodies → less discriminative embeddings). Tightening to 0.29 prevents merging across individuals.

**Final cluster counts:**

| Species | Images | Threshold | Clusters | Avg/cluster |
|---------|--------|-----------|----------|-------------|
| LynxID2025 | 946 | 0.51 | 53 | 17.8 |
| SalamanderID2025 | 689 | 0.29 | 492 | 1.4 |
| SeaTurtleID2022 | 500 | 0.42 | 301 | 1.7 |
| TexasHornedLizards | 274 | 0.30 | 225 | 1.2 |
| **Total** | **2,409** | — | **1,071** | **2.3** |

**Keypoint statistics (test set):**

| Species | SIFT valid | SIFT avg kpts | KAZE valid | KAZE avg kpts |
|---------|-----------|---------------|-----------|---------------|
| LynxID2025 | 945/946 | 395 | 943/946 | 528 |
| SalamanderID2025 | 689/689 | 736 | 689/689 | 794 |
| SeaTurtleID2022 | 468/500 | 507 | 437/500 | 321 |
| TexasHornedLizards | 274/274 | 933 | 274/274 | 924 |

**Key files:** `FINAL_SOLUTION_v2_5/build_notebook_v2_5.py`, `FINAL_SOLUTION_v2_5/ensemble-global-local-reid-v2-5.ipynb` (Kaggle run with outputs), `FINAL_SOLUTION_v2_5/SOLUTION_NOTES.md`

---

### V2.6 — Joint Weight + Threshold Calibration (Score: 0.45498, +1.7%)

**Change from V2.5:** Only the calibration cell was replaced. Everything else is identical.

- V2.5 calibrated: `threshold_cluster` only (global cosine similarity)
- V2.6 calibrates: `global_weight`, `sift_weight`, `kaze_weight`, **and** `threshold_cluster` jointly
  using the full ensemble (global + SIFT match matrix + KAZE match matrix) on training images.

**Calibration method:**
1. Subsample ≤ 500 training images per species
2. Extract MiewID global embeddings (GPU, TTA)
3. Extract SIFT + KAZE descriptors (CPU), compute BFMatcher match matrices with top-50 global preselection
4. Grid-search (gw, sw, thr) with kw = 1 − gw − sw (skip if kw < 0.05), maximise AMI

**Grid:**
- `global_weight` ∈ {0.55, 0.60, 0.65, 0.70, 0.75}
- `sift_weight` ∈ {0.10, 0.15, 0.20, 0.25}
- `kaze_weight` = 1 − gw − sw
- `threshold` ∈ [0.15, 0.59] step 0.02 (23 values)
- ≤ 15 valid weight combos × 23 thresholds = ≤ 345 clustering runs / species

**Actual calibration output (from Kaggle run):**

| Species | Train used | IDs | Old (gw,sw,kw) → New | Old thr → New | AMI | Time |
|---|---|---|---|---|---|---|
| LynxID2025 | 500/2,957 | 61 | (0.70,0.20,0.10)→**(0.70,0.25,0.05)** | 0.35→**0.59** | 0.3315 | 1,102s |
| SalamanderID2025 | 500/1,388 | 318 | (0.65,0.20,0.15)→**(0.55,0.20,0.25)** | 0.35→**0.25** | 0.3923 | **11,992s ⚠️** |
| SeaTurtleID2022 | 500/8,729 | 254 | (0.65,0.20,0.15)→**(0.70,0.15,0.15)** | 0.40→**0.55** | 0.8147 | 184s |
| TexasHornedLizards | — | — | unchanged | 0.30 | — | — |

> ⚠️ Salamander BFMatcher took 11,992s (~3.3h) due to high keypoint density (~794 avg KAZE kpts/image).
> Fix for V2.7: cap calibration keypoints to ~300/image.

**Notable weight shifts:**
- **Salamander**: global 0.65 → 0.55, KAZE 0.15 → 0.25. KAZE provides more discriminative signal for deformable bodies than previously assumed.
- **SeaTurtle**: global 0.65 → 0.70, threshold 0.42 → 0.55 (very permissive merging → fewer, larger clusters → better recall).
- **Lynx**: KAZE weight halved 0.10 → 0.05 (expected — 0% SAM3 coverage means background noise in KAZE); threshold pushed to 0.59 (very permissive).

**Final cluster counts:**

| Species | Images | Threshold | V2.5 clusters | V2.6 clusters | Δ |
|---------|--------|-----------|---------------|---------------|---|
| LynxID2025 | 946 | 0.59 | 53 | **18** | −35 |
| SalamanderID2025 | 689 | 0.25 | 492 | **564** | +72 |
| SeaTurtleID2022 | 500 | 0.55 | 301 | **171** | −130 |
| TexasHornedLizards | 274 | 0.30 | 225 | **225** | 0 |
| **Total** | **2,409** | — | **1,071** | **978** | **−93** |

**Key files:** `FINAL_SOLUTION_v2_6/build_notebook_v2_6.py`, `FINAL_SOLUTION_v2_6/ensemble_global_local_reid_v2_6.ipynb`, `FINAL_SOLUTION_v2_6/SOLUTION_NOTES.md`

---

### V2.7 — XFeat GPU extractor replacing KAZE (Score: 0.37257, −18%)

**Change from V2.6:** Replaced KAZEExtractor with XFeatExtractor (CVPR 2024, GPU-accelerated, 64-dim).

**Why it failed:** XFeat descriptors have a very different match score distribution than KAZE. The calibration found xw=0.05–0.15 (vs KAZE's kw=0.15–0.25), suggesting XFeat is less discriminative for biological texture patterns. Salamander threshold jumped from 0.25 to 0.45 causing massive over-merging (564→206 clusters). KAZE is a gradient-based detector tuned for edge/blob-rich textures exactly like animal skin patterns; XFeat is a learned extractor trained on human-made scenes.

**Key files:** `FINAL_SOLUTION_v2_7/build_notebook_v2_7.py`

---

### V2.8 — Salamander Yellow+Specular Mask (Score: 0.46202, +1.5%) ← CURRENT BEST

**Change from V2.6:** Two changes to `get_seg_mask()` for `SalamanderID2025` only:
1. **Specular highlight removal:** pixels with V>220, S<40 in HSV excluded (flash reflections ~27% of all SIFT kpts)
2. **Yellow-spot ROI:** restrict to H=18-42°, S>80, V>80 in HSV, dilated 25px. Fallback to SAM3 if yellow pixels < 50.

**Motivation:** Fire salamanders are identified by yellow spots on black skin. Flash photography creates specular hotspots (bright, desaturated blobs) that SIFT/KAZE detectors find highly attractive. These account for ~27% of all keypoints but carry zero identity information. Restricting to yellow-spot regions forces descriptors onto the actual discriminative marker.

**Bug fixed:** `_extract_calib_local` was calling `get_seg_mask(key)` with a relative path → `ValueError` in `p.relative_to(_root)` → always returned `None` → SAM3 mask never applied during calibration. Fixed to `get_seg_mask(path)` (absolute). Now calibration and main pipeline use identical features.

**Also added:** `CALIB_KPT_CAP = 300` — caps keypoints/image during calibration only (not main pipeline). Reduced Salamander calibration from 11,992s (V2.6) to 1,550s.

**Calibration output:**

| Species | Old (gw,sw,kw) → New | Old thr → New | AMI | Time |
|---|---|---|---|---|
| LynxID2025 | (0.70,0.20,0.10)→**(0.75,0.10,0.15)** | 0.35→**0.59** | 0.3438 | 973s |
| SalamanderID2025 | (0.65,0.20,0.15)→**(0.65,0.25,0.10)** | 0.35→**0.39** | 0.3614 | **1,550s** ✓ |
| SeaTurtleID2022 | (0.65,0.20,0.15)→**(0.75,0.10,0.15)** | 0.40→**0.51** | 0.8065 | 129s |
| TexasHornedLizards | unchanged | 0.30 | — | — |

**Final cluster counts:**

| Species | Images | Threshold | V2.6 clusters | V2.8 clusters | Δ |
|---------|--------|-----------|---------------|---------------|---|
| LynxID2025 | 946 | 0.59 | 18 | **18** | 0 |
| SalamanderID2025 | 689 | 0.39 | 564 | **334** | −230 |
| SeaTurtleID2022 | 500 | 0.51 | 171 | **215** | +44 |
| TexasHornedLizards | 274 | 0.30 | 225 | **225** | 0 |
| **Total** | **2,409** | — | **978** | **792** | **−186** |

**Key files:** `FINAL_SOLUTION_v2_8/build_notebook_v2_8.py`, `FINAL_SOLUTION_v2_8/ensemble_global_local_reid_v2_8.ipynb`

---

### V3.1 — ALIKED + LightGlue as 3rd Local Component (Score: 0.41500, −10.2%)

**Change from V2.8:** Added ALIKED + LightGlue as a 3rd local feature extractor alongside SIFT and KAZE. Calibration grid expanded to 4 weights: GLOBAL_W + SIFT_W + ALIKED_W (KAZE as residual).

**Architecture addition:**
```
ALIKED-n16 (learned keypoint detector + descriptor, 128-dim)
  → LightGlue matcher (attentional graph neural network)
  → score = 1 - exp(-matches / 20)
```
- `enable_internet=true` in kernel-metadata (LightGlue needs pip install)
- SIFT/KAZE kept GPU ratio test + RANSAC geometric gate from V2.8

**Root cause of regression:** Calibration/test score mismatch. SIFT/KAZE calibration used ratio-only scoring (same as V2.8), but the test pipeline applied a RANSAC geometric gate, producing different score distributions. Calibration assigned kw=0.00 for all species → KAZE effectively disabled at test time → net regression.

**Key lesson:** Calibration and inference MUST use identical score computation paths. If you change the scoring function for inference, you must recompute calibration caches.

**Key files:** `FINAL_SOLUTION_v3_1/build_notebook_v3_1.py`

---

### V3.2 — Ratio-Only SIFT/KAZE + ALIKED Seg Mask Fix (Score: 0.42134, −8.8%)

**Changes from V3.1:**
1. SIFT/KAZE: ratio-only scoring at BOTH calibration and test time (`compute_local_score_ratio_only`) — fixes the calib/test mismatch from V3.1
2. ALIKED: seg_mask passed into `extract()` for keypoint filtering (was missing)
3. Bug fix: `del _k` NameError in ALIKED feature extraction loop
4. Patch 9: recompute `ensemble_similarity_cache` after calibration write-back (critical — V3.1 used stale cache)
5. Cache suffix changed to `_ratio_matches`

**Why still below V2.8:** Grid structure error. Calibration grid had ALIKED explicit but KAZE as residual (kw = 1−gw−sw−aw). Calibration crowded KAZE to kw=0.00 for all species even though KAZE genuinely helps. Threshold ceiling also capped at 0.59 (Lynx true optimum ≈ 0.65).

**Key lesson:** When introducing a new component in a residual grid, all known-good components must be EXPLICIT. Never let a known-good component be the residual.

**Key files:** `FINAL_SOLUTION_v3_2/build_notebook_v3_2.py`, dataset: `svrresearch/animalclef-v3-cache`

---

### V3.2.2 — Threshold Ceiling Fix + KAZE-Explicit Grid (Score: 0.47912, +3.7%) ← NEW BEST

**Changes from V3.2 (applied live in Kaggle environment, same kernel):**
1. Threshold grid extended: `[0.15..0.75, 31 steps]` (was 0.59 ceiling)
2. Grid flipped to KAZE-explicit: `KAZE_W_GRID=[0,0.05,0.10,0.15]`, `aw=1-gw-sw-kw` (ALIKED as residual)
3. THL restored to V2.8 weights: gw=0.75, sw=0.10, kw=0.15, aw=0.00, thr=0.30

**Calibrated weights:**

| Species | gw | sw | kw | aw | thr | clusters | AMI |
|---------|----|----|----|----|-----|---------|-----|
| LynxID2025 | 0.40 | 0.00 | 0.05 | 0.55 | 0.65 | 38/946 | — |
| SalamanderID2025 | 0.50 | 0.30 | 0.15 | 0.05 | 0.47 | 277/689 | — |
| SeaTurtleID2022 | 0.70 | 0.20 | 0.05 | 0.05 | 0.57 | 119/500 | — |
| TexasHornedLizards | 0.55 | 0.15 | 0.10 | 0.20 | 0.30 | 155/274 | — |

**Key insights:**
1. **Threshold ceiling was the bottleneck**: Lynx moved 0.59→0.65, went from 20→38 clusters
2. **KAZE must be explicit in grid**: residual KAZE always gets crowded to 0
3. **ALIKED genuinely useful for Lynx**: aw=0.55 (highest ALIKED weight). Lynx has worst global+SIFT but ALIKED finds meaningful texture patches even without segmentation.

**Key files:** Same kernel as V3.2

---

### V3.2.3 — THL Threshold Override (Score: 0.34934, −27%) — CATASTROPHIC

**Change from V3.2.2:** Extended calibration grids (SIFT up to 0.35, KAZE up to 0.20) — all species returned same weights. THL override: changed thr=0.30→0.40 based on stability sweep "elbow" analysis.

**Root cause:** THL thr=0.40 → 57 clusters. Ground truth THL is ~150-200 individuals from 274 images. 57 clusters = massive over-merging. THL score component collapsed to ~28% of V3.2.2 value.

The stability sweep showed a "steep drop" at thr=0.35-0.40. This was misinterpreted as "good discrimination → right threshold." In reality, the steep drop tells you there IS separation but doesn't tell you WHERE to threshold.

**Key lesson:** THL stability sweep CANNOT reliably predict true cluster count for zero-shot species. THL ALIKED has mean=0.22, std=0.39 (highest std of any species/extractor) — discriminative, do NOT remove. **Keep thr=0.30 → ~155 clusters.**

**Key files:** Same kernel as V3.2.2

---

### V3.3 — SuperPoint + LightGlue as 4th Local Component (Score: 0.47183, −1.5%)

**Change from V3.2.2:** Added SuperPoint + LightGlue as 4th local feature extractor. Baked in V3.2.2 calibrated defaults.

**Calibration results (5-component: MiewID + SIFT + KAZE + ALIKED + SuperPoint):**

| Species | gw | sw | kw | aw | spw | thr | clusters |
|---------|----|----|----|----|-----|-----|---------|
| LynxID2025 | 0.40 | 0.05 | 0.05 | 0.50 | 0.00 | 0.65 | 38/946 |
| SalamanderID2025 | 0.40 | 0.25 | 0.15 | 0.00 | 0.20 | 0.51 | 188/689 |
| SeaTurtleID2022 | 0.70 | 0.10 | 0.00 | 0.10 | 0.10 | 0.63 | 95/500 |
| TexasHornedLizards | 0.55 | 0.15 | 0.10 | 0.20 | 0.00 | 0.30 | 155/274 |

**Root cause:** SuperPoint structural keypoints (corners/junctions) create false positive similarities between different individuals — body outline shapes look similar across animals.
- Salamander: 277→188 clusters (over-merged, spw=0.20)
- SeaTurtle: 119→95 clusters (over-merged, spw=0.10)
- Lynx correctly got spw=0 (ALIKED already handles structure)
- Training AMI improved (over-fitting signal) but test score regressed

**Key lesson:** SuperPoint adds no value for animal re-ID with biological texture patterns. SP detects structural features (corners, junctions) — these are similar across individuals of the same species. Skip SP for future versions.

**Key files:** `FINAL_SOLUTION_v3_3/build_notebook_v3_3.py`

---

### V4.0-ARI — MegaDescriptor-L Dual-Global + ARI Calibration (Score: 0.41769, −12.8%)

**Change from V3.2.2:** Added MegaDescriptor-L-384 as a 2nd global model (separate similarity matrix, NOT concatenated like V2.0). 5-component ensemble: MiewID + MegaDescriptor + SIFT + KAZE + ALIKED. **Used ARI as calibration metric (mistake).**

**Why ARI failed:** ARI (Adjusted Rand Index) rewards large clusters → pushes thresholds too high → over-merging. Lynx: ARI pushed threshold to 0.75 (ceiling) → only **8 clusters** from 946 images (catastrophic). ARI's mathematical structure means that merging two large clusters costs less than splitting them incorrectly, creating a systematic bias toward over-merging.

**Key lesson:** **Always use AMI for calibration, never ARI** — even though the competition metric is ARI. AMI (Adjusted Mutual Information) penalises impure clusters, producing better threshold selection.

**Key files:** Same build as V4.0, different calibration metric

---

### V4.0 — MegaDescriptor-L Dual-Global + AMI Calibration (Score: 0.48600, +1.4%) ← NEW BEST

**Changes from V3.2.2:**
1. Added MegaDescriptor-L-384 as 2nd global model: `timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', num_classes=0)` — 197M params, 1536-dim, 384×384 input, ImageNet normalisation
2. 5-component ensemble: MiewID (mw) + MegaDescriptor (mgw) + SIFT (sw) + KAZE (kw) + ALIKED (aw=residual)
3. Extended calibration grid: MIEW_W=[0..0.50 step 0.10] × MEGA_W=[0..0.50 step 0.10] × SIFT_W=[0..0.40 step 0.05] × KAZE_W=[0..0.20 step 0.05], THR=[0.15..0.75, 31 steps]
4. **AMI metric** (ARI reverted after V4.0-ARI catastrophe)
5. MegaDescriptor embeddings cached separately: `cache/embeddings/{species}_mega.npy`
6. ROOT_DIR fallback: checks `/kaggle/input/animal-clef-2026` then `/kaggle/input/competitions/animal-clef-2026`

**Calibration results:**

| Species | mw | mgw | sw | kw | aw | thr | AMI | Time |
|---------|----|----|----|----|----|----|-----|------|
| LynxID2025 | 0.40 | 0.00 | 0.05 | 0.00 | 0.55 | 0.67 | 0.4823 | 1,449s |
| SalamanderID2025 | 0.40 | 0.00 | 0.40 | 0.15 | 0.05 | 0.51 | 0.4212 | 1,714s |
| SeaTurtleID2022 | 0.50 | 0.30 | 0.10 | 0.10 | 0.00 | 0.65 | 0.9041 | 2,191s |
| THL (uncalibrated) | 0.275 | 0.275 | 0.15 | 0.10 | 0.20 | 0.30 | N/A | — |

**Notable:** Pretrained MegaDescriptor-L useful for SeaTurtle (mgw=0.30) but NOT for Lynx or Salamander (mgw=0.00). MegaDescriptor as SEPARATE sim matrix works — V2.0 failed because of LNBNN scoring, not because of MegaDescriptor itself.

**MegaDescriptor calib takes ~24 min/species** (extracting 384×384 embeddings on GPU). Total calib ~90 min.

**Key files:** `FINAL_SOLUTION_v4_0/build_notebook_v4_0.py`, `FINAL_SOLUTION_v4_0/ensemble_global_local_reid_v4_0.ipynb`, dataset: `sreevaatsavbavana/version-4-cache`

---

### V5.0 — Fine-tuned MegaDescriptor-T-224 + Isotonic + 2-Phase (Score: 0.34825, −28.3%)

**Changes from V4.0:**
1. Fine-tuned MegaDescriptor-T-224 per species using ArcFace loss (via `wildlife_tools.train.BasicTrainer`)
2. Added `IsotonicCalibration` to normalise ensemble scores to [0,1]
3. Added 2-phase clustering: Phase 1 = match test images to known training identities, Phase 2 = cluster remaining unknowns

**Fine-tuning details:**
- Model: MegaDescriptor-T-224 (Swin-T, 27.8M params, 768-dim)
- Loss: ArcFace(margin=0.25, scale=16) — scale=64 causes gradient saturation
- Optimizer: AdamW(lr=1e-4, weight_decay=0.01)
- Epochs: Lynx=10, SeaTurtle=10, Salamander=15
- External data: CzechLynxv2 (~42k images) for Lynx only
- All params trainable from epoch 1, no val split, no early stopping

**Fine-tuning diagnostic results (embedding-only, no ensemble):**

| Species | Pretrained AMI | Fine-tuned AMI | Delta |
|---|---|---|---|
| Lynx | 0.26 | 0.66 | +0.40 |
| SeaTurtle | 0.63 | 0.75 | +0.12 |
| Salamander | 0.17 | 0.23 | +0.06 |

**Root causes of regression (5 diagnosed):**
1. **IsotonicCalibration compressed all score distributions** → thresholds hit 0.75 ceiling for all species
2. **Salamander AMI: 0.42→0.21** (catastrophic). Isotonic degenerate with 587 IDs, few same-individual pairs.
3. **2-phase: threshold_known=0.30 matched ALL images as "known"** (0 unknowns). Too permissive.
4. **Cluster naming bug**: Phase 1 used training identity names (e.g., "lynx_001") instead of `cluster_Species_N` format
5. **Score: 0.34825** — massive regression from V4.0's 0.4860

**Key lesson:** Don't add isotonic+2-phase on top of already-calibrated grid search. V4.0 pure clustering with AMI grid-search was simpler and better.

**Key files:** `FINAL_SOLUTION_v5_0/build_notebook_v5_0.py` (DO NOT USE), `FINAL_SOLUTION_v5_0/finetune_megadescriptor_v5.ipynb` (fine-tuning notebook)

---

### V5.1 — Fine-tuned MegaDescriptor-T-224 + V4.0 Pure Clustering (Score: 0.51705, +6.4%) ← CURRENT BEST

**Changes from V5.0:**
1. **Removed** IsotonicCalibration (caused V5.0 regression)
2. **Removed** 2-phase clustering (caused V5.0 regression)
3. Pure V4.0 clustering with AMI grid-search (same architecture as V4.0 but with fine-tuned model)
4. Extended threshold grid: [0.15..0.85, 36 steps] (was [0.15..0.75, 31 steps] in V4.0)
5. Fine-tuned model loaded per-species; THL falls back to pretrained L-384 (no training data)

**Fine-tuning:** Same as V5.0 (T-224, ArcFace, BasicTrainer). Model unchanged — only the submission notebook changed.

**Calibration results:**

| Species | mw | mgw | sw | kw | aw | thr | clusters | AMI |
|---------|----|----|----|----|----|----|---------|-----|
| LynxID2025 | 0.10 | **0.50** | 0.05 | 0.00 | 0.35 | 0.57 | 25/946 | 0.6898 |
| SalamanderID2025 | 0.40 | 0.00 | 0.40 | 0.15 | 0.05 | 0.51 | 289/689 | 0.4212 |
| SeaTurtleID2022 | 0.50 | 0.20 | 0.15 | 0.10 | 0.05 | 0.63 | 94/500 | 0.8584 |
| THL (uncalibrated) | 0.275 | 0.275 | 0.15 | 0.10 | 0.20 | 0.30 | 201/274 | N/A |

**Key observations vs V4.0:**
- **Lynx**: mgw 0.00→0.50, mw 0.40→0.10. Fine-tuned MegaDesc is now DOMINANT feature. AMI +0.21 (0.48→0.69). CzechLynx external data was key (42k images).
- **Salamander**: Identical to V4.0. Fine-tuned T-224 useless (mgw=0.00). Only 1,388 training images → insufficient for ArcFace.
- **SeaTurtle**: mgw 0.30→0.20 (−0.10). Fine-tuned T-224 WORSE than pretrained L-384 — smaller model capacity (768-dim vs 1536-dim).
- **Thresholds**: None hit ceiling. Extended grid to 0.85 was precautionary.

**Key lessons:**
1. Fine-tuning transforms useless pretrained MegaDesc (Lynx mgw=0.00) into dominant feature (mgw=0.50)
2. CzechLynx external data critical for Lynx (42k images vs 3k competition)
3. T-224 too small for SeaTurtle — need fine-tuned L-384 to get best of both worlds
4. Do NOT add isotonic/2-phase on top of calibrated grid search

**Key files:** `FINAL_SOLUTION_v5_1/build_notebook_v5_1.py`, `FINAL_SOLUTION_v5_1/ensemble_global_local_reid_v5_1.ipynb`, dataset: `sreevaatsavbavana/megadesc-finetuned-v5`

---

### V5.2 — Fine-tuned MegaDescriptor-L-384, Under-trained (Score: 0.41964, −18.8%)

**Change from V5.1:** Fine-tuned MegaDescriptor-L-384 (Swin-L, 197M params, 1536-dim) instead of T-224 (Swin-T, 28M params, 768-dim). Conservative training recipe.

**Training recipe (too conservative):**
```
Model: MegaDescriptor-L-384 (Swin-L, 197M params, 1536-dim)
Loss: ArcFace(margin=0.25, scale=16)
Optimizer: AdamW, lr=3e-5, weight_decay=0.05
Scheduler: CosineAnnealing
Batch: 4 (physical) × 16 (accum) = 64 effective
Grad checkpointing: Yes
Epochs: Lynx=3, SeaTurtle=5, Salamander=10
CzechLynx: 8000 subsample
No LLRD, no warmup
```

**Root cause:** LR=3e-5 was 33× too low for Swin-L-384. Only 3 epochs for Lynx was insufficient. The model barely moved from pretrained weights — essentially paying the compute cost of L-384 without getting the fine-tuning benefit.

**Key lesson:** Large pretrained models (Swin-L) need aggressive LR + layer-wise learning rate decay (LLRD) + warmup for effective fine-tuning. Conservative hyperparameters that worked for T-224 are insufficient.

**Key files:** `FINAL_SOLUTION_v5_2/build_notebook_v5_2.py`, dataset: `sreevaatsavbavana/megadesc-finetuned-v52`

---

### V5.4 — Aggressive L-384 Fine-tuning + Expanded Grids (Score: 0.47917, −7.3%)

**Changes from V5.2:**
1. Completely re-did fine-tuning with aggressive recipe: LR=1e-3 (33× V5.2), LLRD=0.80 across 6 Swin-L stage groups, warmup=10% of steps, cosine decay
2. More epochs: Lynx=6, SeaTurtle=10, Salamander=20
3. Larger CzechLynx subsample: 15,000 (stratified)
4. Larger batch: 36 × 2 (accum) = 72 effective
5. All species use fine-tuned L-384 (not hybrid T-224/L-384)
6. Expanded calibration grids: MEGA_W to 0.80 (was 0.50), THR 0.10-0.90 (was 0.15-0.85)

**LLRD layer groups (6 groups, decay=0.80):**
```
patch_embed: lr × 0.80^5 = 0.328x
layers.0:    lr × 0.80^4 = 0.410x
layers.1:    lr × 0.80^3 = 0.512x
layers.2:    lr × 0.80^2 = 0.640x
layers.3:    lr × 0.80^1 = 0.800x
norm:        lr × 0.80^0 = 1.000x
```

**Diagnostic results (embedding-only, no ensemble):**

| Species | Pretrained AMI | Fine-tuned AMI | Delta |
|---|---|---|---|
| Lynx | 0.3100 | **0.7291** | +0.42 |
| SeaTurtle | 0.8361 | **0.8950** | +0.06 |
| Salamander | 0.2214 | **0.5466** | +0.33 |

Fine-tuned embeddings are objectively excellent — massive improvements over pretrained, especially Lynx (+0.42 AMI) and Salamander (+0.33 AMI).

**Calibration results:**

| Species | mw | mgw | sw | kw | aw | thr | clusters | AMI |
|---------|----|----|----|----|----|----|---------|-----|
| LynxID2025 | 0.00 | **0.80** | 0.00 | 0.05 | 0.15 | 0.50 | **18**/946 | 0.7486 |
| SalamanderID2025 | 0.40 | 0.00 | 0.40 | 0.15 | 0.05 | 0.52 | 274/689 | 0.4246 |
| SeaTurtleID2022 | 0.50 | 0.30 | 0.10 | 0.10 | 0.00 | 0.62 | 107/500 | 0.9046 |
| THL (uncalibrated) | 0.35 | 0.00 | 0.20 | 0.15 | 0.30 | 0.30 | 182/274 | N/A |

**Root causes of regression:**
1. **Lynx: 18 clusters (catastrophic over-merging)** — expanded MEGA_W grid to 0.80 gave calibration too much freedom. mgw=0.80 on training gave excellent AMI (0.7486) but collapsed individuality on test. Fine-tuned embeddings make all lynx look similar → threshold 0.50 merges too aggressively.
2. **THL: lost pretrained MegaDescriptor** — V5.1 had mgw=0.275 → 201 clusters. V5.4 defaults changed to mgw=0.00 → 182 clusters (fewer, less accurate).

**Key lessons:**
1. **Expanded calibration grids can hurt**: too much freedom → over-fitting to training distribution
2. **Training AMI ≠ test performance**: Lynx AMI 0.7486 on train → 18 clusters on test
3. Fine-tuned model quality is excellent but **CALIBRATION is the bottleneck** — the model improvement doesn't translate to score improvement without proper weight constraints

**Key files:** `FINAL_SOLUTION_v5_4/build_notebook_v5_4.py`, `FINAL_SOLUTION_v5_4/ensemble_global_local_reid_v5_4.ipynb`, dataset: `svrresearch/megadesc-v54-finetune`

---

### V5.5 — Hotfix: Fixed Weights + Threshold-Only Calibration (Score: 0.50506, −2.3%)

**Changes from V5.4:** No weight grid search. Manually picked weights based on V5.1/V5.4 analysis, then calibrated thresholds only (41 steps per species).

**Weight choices and rationale:**

| Species | mw | mgw | sw | kw | aw | Rationale |
|---|---|---|---|---|---|---|
| Lynx | 0.05 | **0.60** | 0.05 | 0.05 | 0.25 | mgw capped between V5.1 (0.50) and V5.4 (0.80) |
| Salamander | 0.30 | **0.20** | 0.30 | 0.15 | 0.05 | Gave fine-tuned MegaDesc a shot (diagnostic AMI 0.55) |
| SeaTurtle | 0.50 | 0.30 | 0.10 | 0.10 | 0.00 | Kept V5.4 calibrated (AMI=0.9046, excellent) |
| THL | 0.275 | **0.275** | 0.15 | 0.10 | 0.20 | Restored V5.1 pretrained MegaDesc defaults |

**Calibration results (threshold-only):**

| Species | thr (default) | thr (calibrated) | AMI | clusters |
|---|---|---|---|---|
| LynxID2025 | 0.55 | 0.50 | 0.7360 | 44/946 |
| SalamanderID2025 | 0.52 | 0.48 | 0.4041 | 319/689 |
| SeaTurtleID2022 | 0.62 | 0.62 | 0.9046 | 107/500 |
| THL (unchanged) | 0.30 | 0.30 | N/A | 201/274 |

**What helped:**
- THL restored to 201 clusters (matched V5.1 exactly — the fix worked)
- SeaTurtle unchanged (already good from V5.4)

**What hurt:**
- **Lynx: 44 clusters (over-fragmented)** — V5.1 had 25. mgw=0.60 with thr=0.50 splits too much on test.
- **Salamander: AMI dropped 0.4212→0.4041** — adding mgw=0.20 hurt. Three prior calibrations (V4.0, V5.1, V5.4) all chose mgw=0.00. Manual override confirmed mgw=0.00 is definitively correct for Salamander in ensemble.

**Key lessons:**
1. **Salamander mgw=0.00 is definitively correct in ensemble**: 4 independent tests agree. Standalone AMI 0.55 does NOT mean it helps the ensemble (MiewID+SIFT already capture the signal; MegaDesc adds correlated noise).
2. **Lynx cluster count highly sensitive**: 18 (V5.4, over-merged) → 25 (V5.1, sweet spot) → 44 (V5.5, over-fragmented). True Lynx test count likely ~25-30.
3. **Threshold-only calibration is fast (~1 min)** vs full grid search (~90 min) but can't recover from wrong manual weights.

**Key files:** `FINAL_SOLUTION_v5_5/build_notebook_v5_5.py`, `FINAL_SOLUTION_v5_5/ensemble_global_local_reid_v5_5.ipynb`

---

### V5.6 — HDBSCAN Clustering + V5.4 L-384 Model (Score: 0.21434, −58.5% from best)

**Changes from V5.1:**
1. Replaced `AgglomerativeClustering` with `sklearn.cluster.HDBSCAN` (metric=precomputed)
2. Switched from V5.1 fine-tuned T-224 to V5.4 fine-tuned L-384 model (`megadesc-v54-finetune`)
3. Fixed V5.1 weights (no weight grid search); calibration only searched HDBSCAN params
4. HDBSCAN grid: `mcs=[2,3,4]`, `ms=[1,2]`, `eps=linspace(0.1,0.7,13)`, `method=["eom","leaf"]` = 156 combos/species
5. Noise points (-1) → singleton clusters (unique IDs)
6. Datasets: `animalclef-v52-cache` (replaces v3-cache), `megadesc-v54-finetune`

**Hypothesis:** HDBSCAN's density-based clustering handles variable cluster sizes better than a single global threshold.

**Calibrated HDBSCAN params & test results:**
| Species | mcs | ms | eps | method | Calib AMI | Test clusters | Noise singletons |
|---|---|---|---|---|---|---|---|
| LynxID2025 | 3 | 1 | 0.200 | eom | 0.6109 | 86/946 | 80/86 = **93%** |
| SalamanderID2025 | 2 | 1 | 0.250 | eom | 0.4016 | 504/689 | 435/504 = **86%** |
| SeaTurtleID2022 | 2 | 1 | 0.450 | leaf | 0.8592 | 162/500 | 60/162 = 37% |
| THL (default) | 2 | 1 | 0.300 | eom | N/A | 112/274 | 89/112 = 79% |

Compare to V5.1 (AgglomerativeClustering): Lynx 25, Salamander 289, SeaTurtle 94, THL 201.

**Score: 0.21434** — catastrophic regression from V5.1 (0.51705).

**Root cause — HDBSCAN noise singletons:** HDBSCAN marks low-density points as noise (-1), each of which becomes its own cluster. Animal re-ID has long-tail identity distributions (many identities with 1-2 images) → inherently sparse density → HDBSCAN marks most images as noise → massive over-fragmentation.
- Salamander: 587 IDs across only 1388 training images (avg 2.4 imgs/ID) → 63% of test images marked noise
- Calibration AMI on training looked decent but did not predict test-time fragmentation
- AgglomerativeClustering never creates noise — every image gets merged → always better for sparse re-ID data

**Build-time issues:**
- ALIKED LightGlue calibration is sequential (1 pair/call) → Lynx with 2957 images = 73k pairs = 2+ hrs. Capped at 600 imgs.
- `animalclef-v52-cache` does NOT contain pre-computed calib matrices (`*_sift_matrix.npy` etc.) → everything recomputed from scratch each run
- SIFT took ~40 min, KAZE ~40 min, ALIKED 2+ hrs for Lynx (3+ hrs before interrupting and capping images)

**Lesson: HDBSCAN is definitively wrong for animal re-ID. Do not try again.**

**Key files:** `FINAL_SOLUTION_v5_6/build_notebook_v5_6.py`, `FINAL_SOLUTION_v5_6/ensemble_global_local_reid_v5_6.ipynb`

---

### V5.7 — k-Reciprocal Re-ranking (Score: 0.49050, −5.1% from best)

**Changes from V5.1:**
1. Added k-Reciprocal Re-ranking (Zhong et al., CVPR 2017) as post-processing
2. Applied after ensemble similarity computation, before clustering
3. Two-stage calibration: Stage 1 = weight grid search on raw cosine, Stage 2 = threshold search on re-ranked matrix
4. Parameters: K1=20, K2=6, lambda=0.3 (from Paper 253, 2nd place AnimalCLEF 2025)
5. THL skipped from re-ranking (no training data for threshold calibration)
6. Same 4 datasets as V5.1 (no new models or features)

**Hypothesis:** Suppressing false matches via reciprocal neighbor verification should improve clustering purity.

**Bugs found during development (3 critical):**
1. **Jaccard formula wrong**: Used `dot / (2 - dot)` which assumes L2-normalized vectors, but code L1-normalizes. Denominator was ~2 instead of ~0.15, crushing Jaccard to ~0.025 instead of ~0.33. Fixed with `norms_sq` computation.
2. **Stage 2 threshold not saved**: `calibrated_config[sp]` populated before Stage 2 ran. Clustering used Stage 1 threshold (cosine scale) on Jaccard-scale similarities → all singletons. Fixed with write-back.
3. **THL re-ranked without calibration**: Re-ranking applied to all species but THL threshold (0.30) set for raw cosine → all singletons. Fixed with `CALIB_SPECIES` gate.

**Score: 0.49050** — regression from V5.1 (0.51705).

**Root cause:** k-Reciprocal Re-ranking is designed for single-query retrieval (1 query vs N gallery), not symmetric all-pairs clustering with mixed ensemble similarities. The reciprocal neighbor verification adds noise to our already-calibrated ensemble rather than improving it. The weighted combination (0.7 Jaccard + 0.3 cosine) dilutes the original discriminative signal without providing enough compensating benefit.

**Lesson: k-Reciprocal Re-ranking does not help our multi-feature ensemble pipeline. Post-processing re-ranking is not a substitute for better features. Do not try again.**

**Key files:** `FINAL_SOLUTION_v5_7/build_notebook_v5_7.py`, `FINAL_SOLUTION_v5_7/ensemble_global_local_reid_v5_7.ipynb`

---

### V5.8 — Fine-Tuning Infrastructure (no direct submission)

**Work done (2026-03-06 to 2026-03-09):**
- Fine-tuned **MegaDescriptor-L-384** (Swin-L, 1536-dim) with ArcFace, LLRD, gradient accumulation
  - Config: BATCH_SIZE=24, ACCUM_STEPS=3 (effective batch 72), BASE_LR=1e-3, LLRD_DECAY=0.80, WARMUP_RATIO=0.10
  - Species epochs: Lynx=6, SeaTurtle=10, Salamander=20
  - Lynx: CzechLynx 15k subsample + competition 3k
  - Diagnostic standalone AMIs: Lynx PT=0.0693→FT=0.5623, SeaTurtle PT=0.8134→FT=0.9228, Salamander PT=0.0986→FT=0.1543
- Fine-tuned **MiewID v3** (EfficientNetV2-RW-M, 2152-dim) with ArcFace, LLRD
  - Config: BATCH_SIZE=16, ACCUM_STEPS=4 (effective batch 64), BASE_LR=5e-4, LLRD_DECAY=0.85
  - Species epochs: Lynx=10, SeaTurtle=15, Salamander=25
  - Diagnostic standalone ARIs: Lynx PT=0.0953→FT=0.8928, SeaTurtle PT=0.7248→FT=0.9745, Salamander PT=0.1184→FT=0.3484
  - Critical loading bug: Fine-tuned weights saved from `MiewIdNet.backbone.state_dict()` (EfficientNet keys); must load into `model.backbone.backbone`, NOT `model.backbone`
- Salamander chain fine-tuning from SeaTurtle weights: PT ARI=0.1152, FT v58=0.3258, Chain v58b=0.3265 (marginal improvement)

**Datasets uploaded to Kaggle:** `sreevaatsavbavana/megadesc-finetuned-v58`, `svrresearch/miewid-finetuned-v58`

**Key files:** `FINAL_SOLUTION_v5_8/build_finetune_v5_8.py`, `FINAL_SOLUTION_v5_8/build_finetune_miewid_v5_8.py`, `FINAL_SOLUTION_v5_8/build_finetune_miewid_salamander_v5_8.py`

---

### V5.9 — Fine-Tuned MiewID + Fine-Tuned MegaDesc-L + TTA (Score: 0.49132, −4.8% from best)

**Changes from V5.1:**
1. Fine-tuned MiewID v3 (V5.8 weights) for Lynx, SeaTurtle, Salamander
2. Fine-tuned MegaDescriptor-L-384 (V5.8 weights) for Lynx, SeaTurtle, Salamander; pretrained L-384 for THL
3. Multi-crop TTA: 5-crop + hflip (use_multicrop=True)
4. Cache preloaded from `sreevaatsavbavana/animalclef-v59-cache` dataset to skip 2.5hr extraction

**Calibration results (hardcoded into notebook after 2.5hr extraction):**
| Species | mw | mgw | sw | kw | aw | thr | ARI |
|---------|----|----|----|----|-----|-----|-----|
| Lynx | 0.70 | 0.30 | 0.00 | 0.00 | 0.00 | 0.47 | 0.9705 |
| SeaTurtle | 0.80 | 0.20 | 0.00 | 0.00 | 0.00 | 0.47 | 0.9706 |
| Salamander | 0.40 | 0.00 | 0.40 | 0.05 | 0.15 | 0.59 | 0.3191 |
| THL | 0.275 | 0.275 | 0.15 | 0.10 | 0.20 | 0.30 | N/A |

**Score: 0.49132** — regression from V5.1 (0.51705).

**Root causes:**
1. **Memorisation**: Fine-tuned MiewID achieves 0.89 training ARI for Lynx but hurts test generalisation. The model learns training-identity-specific features that don't transfer to unseen test individuals.
2. **Calibration gave 0 weight to local features for Lynx/SeaTurtle** (aw=0.00, sw=0.00, kw=0.00) — in V5.1, ALIKED had aw=0.35 for Lynx. Fine-tuning dominated so strongly it masked the complementary local signal.
3. **Multi-crop TTA** degrades fine-tuned model performance — crops average out the sharp discriminative features learned during fine-tuning.

**Bugs encountered during development:**
- MiewID loading key mismatch (see V5.8 for fix details)
- `NameError: calib_model is not defined` — `del calib_model` inside loop followed by `del calib_model` after loop. Fixed with `calib_model = None`.
- Cache path: dataset mounted at `/kaggle/input/datasets/.../animalclef-v59-cache/cache/` (extra subfolder)
- ROOT_DIR: must be `/kaggle/input/competitions/animal-clef-2026` (not `/kaggle/input/animal-clef-2026`)

**Key files:** `FINAL_SOLUTION_v5_8/build_notebook_v5_9.py`, `FINAL_SOLUTION_v5_8/ensemble_global_local_reid_v5_9.ipynb`

---

### V5.9b — Fine-Tuned MiewID + Fine-Tuned MegaDesc-L, No TTA (Score: 0.48379, −6.4% from best)

**Changes from V5.9:** Disabled multi-crop TTA (use_multicrop=False). All embeddings are single-pass (original image only). Cache suffix changed to `_v59b`.

**Score: 0.48379** — worse than V5.9 (0.49132) and V5.1 (0.51705).

**Root cause:** Removing TTA made the fine-tuned models even worse. The issue is not TTA — it's the underlying memorisation problem. Fine-tuned MiewID and MegaDesc achieve 90%+ training ARI but generalise poorly to completely unseen test identities.

**Lesson learned (critical):** Fine-tuning a model on in-distribution training identities does NOT guarantee better features for out-of-distribution test identities. The model can overfit to the specific 61/350/469 training individuals, losing the generalisation of the pretrained foundation model. For wildlife re-ID with small per-species training sets, pretrained models can be superior to fine-tuned ones.

**Key files:** `FINAL_SOLUTION_v5_8/build_notebook_v5_9.py` (rebuild with use_multicrop=False)

---

## Track B — Ambitious Experiments (all regressed or unconfirmed)

### V3 — MegaDescriptor + WildFusion Isotonic Calibration (Score: 0.15812, −58%)

**Approach:** Replace MiewID v3 with MegaDescriptor-L-384. Add isotonic regression calibration. Multi-extractor local matching (SIFT, SuperPoint, ALIKED, DISK).

**Architecture:**
```
Image → MegaDescriptor-L-384 (Swin-L, 1536-dim) → cosine sim → isotonic calibration → [0,1]
      → Local features (SIFT / SuperPoint / ALIKED / DISK) → matches → isotonic calibration → [0,1]
                                                → Weighted fusion → Known/Unknown → clusters
```

**Submitted as:** `version-3.ipynb` (585 unique clusters from 2409 images)

**Validation outputs:**
- SeaTurtle: AMI best at global=0.60, threshold=0.45
- Lynx: best at global=0.35, threshold=0.45
- Salamander: 78 clusters / 56 val identities (AMI=0.456)
- Test: Turtle 124 matched/376 new, Lynx 67 matched/879 new, Salamander 72 matched/617 new

**Root causes of failure (5 diagnosed in V3_FAILURE_ANALYSIS.md):**

1. **Wrong backbone:** MegaDescriptor top-1 accuracy 59.2% vs MiewID 78.4% (−19.2 pp). The whole approach was built on a weaker model.

2. **Wrong metric — AMI instead of ARI:** AMI increases monotonically as you add more clusters (every image in its own cluster = perfect AMI). Grid-searching on AMI → selects highest threshold → maximum fragmentation → all singletons → ARI ≈ 0.15.

3. **Isotonic calibration collapsed:** `<2%` positive pairs in validation split → calibrator maps all similarities to ~0 (insufficient positive signal). `IsotonicRegression(out_of_bounds='clip')` needs balanced positive/negative pairs.

4. **Global threshold, not per-species:** V3 used a single threshold=0.50 for all species. Different species have completely different similarity distributions.

5. **Ensemble complexity without validated components:** 5-model pipeline with wrong backbone, wrong metric, wrong calibration, wrong threshold — impossible to debug.

**Evidence:** Submission had 2409 rows, ~2200 unique clusters (near-all singletons), match_rate=0.12, avg_cluster_size=1.1.

**Key files:** `FINAL_SOLUTION_v3/ensemble_wildlife_tools_v3.ipynb`, `FINAL_SOLUTION_v3/version-3.ipynb`, `FINAL_SOLUTION_v4/V3_FAILURE_ANALYSIS.md`

---

### V4 — MiewID v3 + k-Reciprocal Re-ranking + ARI Grid Search

**Two submissions with different outcomes.**

**Motivation:** Fix V3's 5 root causes: return to MiewID v3, switch metric to ARI, add k-Reciprocal Re-ranking (Zhong et al. CVPR 2017), per-species threshold grid-search on training data.

**k-Reciprocal Re-ranking (Zhong et al. 2017):**
```
For pair (i,j): Jaccard similarity = |R(i,k) ∩ R(j,k)| / |R(i,k) ∪ R(j,k)|
Final distance = (1-λ) * original_distance + λ * (1 - Jaccard_similarity)
```
Validation ARI gains: SeaTurtle 0.084 → 0.796, Salamander 0.001 → 0.199 (but Lynx 0.000 → 0.050 — k-RR hurts Lynx on val).

**Architecture:**
```
MiewID v3 (2152-dim) → cosine sim → k-Reciprocal Re-ranking
DISK + ALIKED (SeaTurtle only) → local match scores
→ Weighted ensemble → ARI-optimal threshold (per-species grid search on train split)
→ AgglomerativeClustering
```

#### V4 Run 1 (Score: 0.27624)

**Issue:** Threshold range ceiling — all species selected maximum values (threshold=0.675, λ=0.40). Lynx produced 317 clusters for 899 val images (over-split). The optimal threshold was above the search ceiling.

#### V4 Run 2 (Score: 0.15913)

**Issue:** Extended range to [0.10, 0.95] → Lynx grid search selected threshold=0.850. Looked fine on validation (23 clusters for val Lynx) but on test with unknown individuals (fewer images/identity → sparser k-RR → Jaccard leaks → everything merged):
- Lynx: entire test set collapsed to 1–3 mega-clusters → ARI collapses
- True optimal Lynx threshold: ~0.745–0.780 (but unreachable from val signal alone)

**Key insight from V4:** k-Reciprocal Re-ranking is unreliable for re-ID datasets with very few images per identity on the test set. The Jaccard similarity (shared reciprocal neighbors) requires multiple images per identity to be non-zero. With 2–5 test images per individual, Jaccard ≈ 0.06–0.13 → re-ranked distance floor ≈ 0.61 >> clustering thresholds → structural failure mode identical to LNBNN in V2.0.

**Key files:** `FINAL_SOLUTION_v4/ensemble_miewid_v4.ipynb`, `FINAL_SOLUTION_v4/README.md`

---

### V5 — Dual-Backbone ArcFace Projection Heads (Score: ~0.27, unconfirmed)

**Motivation:** V4's 0.27624 bottleneck = Lynx (train ARI 0.046) and Salamander (val ARI 0.199). MiewID's generic embeddings lack discriminability for these species. Inspired by 2nd-place AnimalCLEF 2025 solution (67.42%).

**Architecture:**
```
Image
  ├── MiewID v3 (frozen, 2152-dim)
  └── MegaDescriptor-L-384 (frozen, 1536-dim)
         concat → 3584-dim
              → Linear(3584, 512) + BN + ReLU + Dropout(0.1)
              → Linear(512, 256) + BN + L2-normalize
              → 256-dim species-specific embedding
              → ArcFace Loss (scale=64, margin per species)
```

**Training on cached embeddings (not raw images):**
- Backbone embeddings extracted once and cached to `.npy`
- Only the projection head (≈2M params) trained → ~10 minutes on T4 GPU
- Gaussian noise std=0.01 added to embeddings during training (augmentation substitute)
- `WeightedRandomSampler` for class balance
- `GroupShuffleSplit(groups=identity)` ensures no identity leakage into validation

**ArcFace parameters per species:**

| Species | Identities | Margin | Epochs | LR |
|---------|-----------|--------|--------|-----|
| SeaTurtleID2022 | ~200 (val) | 0.5 rad | 15 | 5e-4 |
| LynxID2025 | 77 | 0.3 rad | 30 | 1e-3 |
| SalamanderID2025 | 587 | 0.5 rad | 40 | 1e-3 |
| TexasHornedLizards | 0 (zero-shot) | — | — | — |

**Safety:** Auto-revert if fine-tuned ARI < MiewID baseline by >0.01; Lynx threshold capped at 0.760.

**Submitted version (version5-0.ipynb — labeled as V6 internally):**
- Val ARI after k-RR: Turtle=0.9171, Lynx=0.0066 (k-RR hurt Lynx), Salamander=0.2192
- Grid search best: Turtle ARI=0.8646 (thresh=0.575), Lynx ARI=0.9733 (thresh=0.900 — suspicious)
- Test clusters: Turtle=193, Lynx=653, Salamander=261, THL=71 → total 1,178 clusters
- Leaderboard score: estimated ~0.27 (not confirmed — notebook labeled as V6)

**Key concern:** Lynx val ARI=0.9733 at threshold=0.900 likely overfit to the validation split (same generalization failure as V4 Run 2). Lynx collapsed on test again (653 clusters vs V2.5's 53).

**Key files:** `FINAL_SOLUTION_v5/ensemble_miewid_v5.ipynb`, `FINAL_SOLUTION_v5/version5-0.ipynb`, `FINAL_SOLUTION_v5/docs/approach.md`, `FINAL_SOLUTION_v5/docs/arcface_for_clustering.md`

---

### V6 — Multi-View Augmented Extraction + Enhanced ArcFace (unconfirmed)

**Approach:** Same dual-backbone architecture as V5 but with:
- Multi-view TTA during embedding extraction (multiple crops + flips)
- Enhanced ArcFace with sub-center variant (more robust to noisy labels)

**Status:** Notebook drafted (`FINAL_SOLUTION_v6/ensemble_miewid_v6.ipynb`), no saved outputs. Score not confirmed on leaderboard. The `version5-0.ipynb` (which was labeled V6 internally) is the closest executed version — see V5 entry above.

**Key files:** `FINAL_SOLUTION_v6/ensemble_miewid_v6.ipynb`

---

### V7 — 8-Component Pipeline (not submitted)

**Approach (most ambitious):** 8 simultaneous architectural changes:
1. **HDBSCAN + Consensus Clustering** — density-based clustering instead of agglomerative
2. **Sub-Center ArcFace** — handles inter-class variation within one identity
3. **Orientation-Aware Similarity** — different weights per image orientation pair
4. **BioCLIP 2** as 3rd backbone — specialised for wildlife imagery; 768-dim
5. **Segmentation Preprocessing** — SAM2 for Lynx (fixing 0% coverage)
6. **Powell-Optimised Backbone Weights** — scipy.optimize.minimize instead of manual weights
7. **Pseudo-Label Self-Training for THL** — iterative refinement since THL has no training identities
8. **Lynx Hardening** — special treatment for the problematic species

**Combined embedding:** MiewID v3 (2152-d) + MegaDescriptor (1536-d) + BioCLIP 2 (768-d) → 4456-d

**Status:** Notebook drafted (`FINAL_SOLUTION_v7/ensemble_miewid_v7.ipynb`), no outputs saved, not submitted. The V6 best noted in the notebook as ~0.27–0.30.

**Key files:** `FINAL_SOLUTION_v7/ensemble_miewid_v7.ipynb`, `FINAL_SOLUTION_v7/build_notebook.py`

---

## Cross-Track Lessons Learned

### 1. Model selection matters most
MiewID v3 (78.4% top-1) consistently outperforms pretrained MegaDescriptor-L-384 (59.2% top-1) as standalone model. But **fine-tuned** MegaDescriptor is transformative: V5.1 Lynx went from mgw=0.00 (pretrained useless) to mgw=0.50 (fine-tuned dominant, AMI 0.48→0.69).

### 2. AMI for calibration, not ARI
V3(B)'s grid search on AMI → all singletons. V4.0-ARI's grid search on ARI → Lynx collapsed to 8 clusters. The resolution: **use AMI as calibration metric** (penalises impure clusters) even though the competition metric is ARI. AMI selects thresholds that generalise; ARI systematically over-merges.

### 3. k-Reciprocal Re-ranking fails for sparse test sets
With ≤5 test images per identity, Jaccard similarity ≈ 0.06–0.13. Re-ranked distance floor ≈ 0.61 >> clustering thresholds 0.30–0.40. Fundamental structural failure — applies equally to any dataset with <10 images per identity.

### 4. LNBNN fails for large descriptor pools
With K=100 candidates × ~1000 keypoints = ~100k descriptors in pool, the LNBNN background distance collapses to ~0.10–0.15 Hellinger → score = 0 for most pairs → match matrix = identity. Only usable when pool size < ~10k descriptors.

### 5. SAM3 mask filtering (not image replacement)
Run SIFT/KAZE on original image, discard keypoints at masked pixels. Do NOT paint background before extraction → boundary artifact keypoints destroy matching.

### 6. Threshold calibration from training labels is the single highest-leverage lever
All Track A thresholds were hand-tuned until V2.5. AMI grid-search on training identities found that Lynx threshold was off by +0.16 (0.35 → 0.51), causing ~6× over-splitting. This single parameter change accounts for most of the +0.070 V2.3c→V2.5 gain.

### 7. KAZE consistently complements SIFT
Non-linear anisotropic diffusion scale space finds different stable keypoints than Gaussian pyramid SIFT. Consistent improvement across all species, including Lynx with no SAM3 masking (KAZE mean score 0.086 vs SIFT 0.057). Always add KAZE alongside SIFT.

### 8. RootSIFT is a free improvement over raw SIFT
L1-normalize + sqrt converts L2 distance to Hellinger distance, which is more discriminative for gradient-histogram descriptors. Two numpy lines, zero extra compute, +12.5% improvement (V1.3 → V2.2). Do not use RootSIFT on KAZE (signed M-SURF descriptors → NaN).

### 9. Do not post-process already-optimal embeddings
MiewID v3 produces near-optimal embeddings for this task. Alpha-QE over-smoothed them (V2.4: −5.6%). ArcFace projection heads (V5(B)) likely helped SeaTurtle but over-merged Lynx. The embedding space needs calibration at the threshold level, not the embedding level.

### 10. Add one change per submission
V2.0 added MegaDescriptor + RootSIFT + LNBNN simultaneously. When it failed, the root cause required 2 diagnostic submissions (V2.1) to identify. Every subsequent Track A submission changed exactly one thing.

### 11. Calibration/test pipeline must be identical
V3.1 used RANSAC at test time but ratio-only during calibration → KAZE weights pushed to 0 → regression. V3.2 fixed this by using ratio-only in both. Any scoring function difference between calibration and inference will distort weights.

### 12. Calibration grid size: more freedom ≠ better results
V5.4 expanded MEGA_W to 0.80 (was 0.50). Calibration picked mgw=0.80 for Lynx with excellent training AMI (0.7486) but catastrophic test clustering (18 clusters). **Keep calibration grids conservative** — V5.1's MEGA_W [0..0.50] was safer.

### 13. Training AMI ≠ test performance
High training AMI can mean over-fitting to training identity distribution. Lynx V5.4: AMI=0.7486 on train → 18 clusters on test (catastrophic). The model learns training-specific patterns that don't generalise.

### 14. Standalone feature quality ≠ ensemble value
V5.4 fine-tuned MegaDescriptor: Salamander standalone AMI=0.5466 (excellent). But in ensemble with MiewID+SIFT+KAZE, adding mgw=0.20 HURT (AMI 0.42→0.40). Four independent calibrations all agree mgw=0.00 for Salamander. Correlated/redundant features don't help.

### 15. SuperPoint fails for animal re-ID
SP structural keypoints (corners/junctions) create false positive similarities between different individuals — body outline shapes look similar across animals. Sal 277→188 clusters, Turtle 119→95. Skip SP.

### 16. Fine-tuning recipe matters enormously for large models
V5.2 (LR=3e-5, no LLRD) = under-trained → 0.41964. V5.4 (LR=1e-3, LLRD=0.80, warmup) = properly trained → diagnostic AMIs excellent. 33× LR increase + LLRD + warmup made the difference for Swin-L-384.

---

## Full Score Timeline

```
Track A:                                            Track B (early):
V1    0.30655 ──────────────────────────────────── Baseline
V1.2  0.26330 ▼ SAM3 white-bg (wrong approach)
V1.3  0.32528 ▲ SAM3 mask filter (correct)          V3(B) 0.15812 ▼ MegaDesc+AMI+isotonic
V2.0  0.10691 ▼ +MegaDesc+RootSIFT+LNBNN            V4(B) 0.27624 ▲ MiewID+kRR+ARI
V2.1  0.10957 ─ Diagnostic (LNBNN confirmed)         V4(B) 0.15913 ▼ Extended range→Lynx over-merge
V2.2  0.36607 ▲ RootSIFT only (+12.5%)               V5(B) ~0.27   ? ArcFace projection heads
V2.3a 0.36378 ▼ Uniform weights                      V6(B) ~0.27   ? Multi-view ArcFace
V2.3b 0.36937 ▲ KAZE for SeaTurtle                   V7(B) ------  Not submitted
V2.3c 0.37726 ▲ KAZE for THL
V2.4  0.35624 ▼ Alpha-QE (over-smoothed)
V2.5  0.44747 ▲ KAZE all species + threshold calib
V2.6  0.45498 ▲ Joint weight + threshold calib
V2.7  0.37257 ▼ XFeat replace KAZE
V2.8  0.46202 ▲ Salamander yellow+specular mask
V3.1  0.41500 ▼ ALIKED+LightGlue (calib/test mismatch)
V3.2  0.42134 ▲ ratio-only SIFT/KAZE + ALIKED fix
V3.2.2 0.47912 ▲ threshold ceiling fix + KAZE-explicit ← prev best
V3.2.3 0.34934 ▼ THL thr=0.40 → catastrophic
V3.3  0.47183 ▼ SuperPoint (false positives)
V4.0A 0.41769 ▼ MegaDesc-L dual-global + ARI calib (over-merge)
V4.0  0.48600 ▲ MegaDesc-L dual-global + AMI calib ← prev best
V5.0  0.34825 ▼ Fine-tuned T-224 + isotonic + 2-phase
V5.1  0.51705 ▲ Fine-tuned T-224 + V4.0 pure clustering ← CURRENT BEST
V5.2  0.41964 ▼ Fine-tuned L-384 (under-trained)
V5.4  0.47917 ▼ Fine-tuned L-384 (over-fit calib)
V5.5  0.50506 ▲ V5.4 hotfix (fixed weights, thr-only calib)
V5.6  0.21434 ▼ HDBSCAN + V5.4 L-384 — catastrophic (noise singletons)
V5.7  0.49050 ▼ k-Reciprocal Re-ranking — regression (dilutes ensemble signal)
V5.9  0.49132 ▼ Fine-tuned MiewID+MegaDesc+TTA — regression (memorisation)
V5.9b 0.48379 ▼ Fine-tuned MiewID+MegaDesc no TTA — worse than V5.9
```

---

## Current Pipeline (V5.1) — Quick Reference

```
Test images
    │
    ├─────────────────────────────────────────────────────────────┐
    ▼                                                             ▼
MiewID v3 (EfficientNetV2-RW-M, 2152-dim)          Fine-tuned MegaDescriptor-T-224 (768-dim)
    + TTA (original + hflip) → L2 normalize              or pretrained MegaDescriptor-L-384 (1536-dim, THL only)
    │                                                     + TTA (original + hflip) → L2 normalize
    │                                                     │
    ├── mw × cosine_sim(MiewID)                           ├── mgw × cosine_sim(MegaDesc)
    │                                                     │
    ├── RootSIFT (SAM3+yellow+specular-masked for Salamander; SAM3-masked others)
    │       └── BFMatcher (ratio=0.75, K=100 GPU candidates)
    │               └── sw × (1 - exp(-matches/20))
    │
    ├── KAZE (SAM3+yellow+specular-masked for Salamander; SAM3-masked others; Lynx on original)
    │       └── BFMatcher (ratio=0.75, K=100 GPU candidates)
    │               └── kw × (1 - exp(-matches/20))
    │
    └── ALIKED-n16 + LightGlue
            └── aw × (1 - exp(-matches/20))
    │
    ▼
Ensemble = mw×MiewID + mgw×MegaDesc + sw×SIFT + kw×KAZE + aw×ALIKED  (weights sum to 1.0)
    │
    ▼
AMI-calibrated weights + thresholds (grid search on training identities)
    Lynx:       mw=0.10, mgw=0.50, sw=0.05, kw=0.00, aw=0.35, thr=0.57
    Salamander: mw=0.40, mgw=0.00, sw=0.40, kw=0.15, aw=0.05, thr=0.51
    SeaTurtle:  mw=0.50, mgw=0.20, sw=0.15, kw=0.10, aw=0.05, thr=0.63
    THL:        mw=0.275, mgw=0.275, sw=0.15, kw=0.10, aw=0.20, thr=0.30 (uncalibrated)
    │
    ▼
AgglomerativeClustering (average linkage, precomputed distance)
    │
    ▼
609 clusters from 2,409 test images → submission.csv
    (Lynx: 25, Sal: 289, Sea: 94, THL: 201)
```

---

## What Has Been Tried (Summary)

| Technique | Tried? | Result | Notes |
|-----------|--------|--------|-------|
| k-Reciprocal Re-ranking | V5.7 | Regression (0.49050) | Dilutes ensemble signal; not suitable for symmetric all-pairs |
| HDBSCAN clustering | V5.6 | Catastrophic (0.21434) | Noise class = all singletons; min_samples tuning crucial |
| Fine-tune MegaDescriptor-T-224 | V5.1 | Best (0.51705) | CzechLynx crucial |
| Fine-tune MegaDescriptor-L-384 | V5.4/V5.9 | Regression | Calibration over-fit + memorisation |
| Fine-tune MiewID v3 | V5.9/V5.9b | Regression | Memorisation — training ARI 0.89 → test regression |
| Multi-crop TTA | V5.9 | Marginal help vs no-TTA | 0.49132 vs 0.48379; both below V5.1 |
| Alpha-QE query expansion | V2.4 | Regression | MiewID embeddings need no post-processing |
| SuperPoint + LightGlue | V3.3 | Regression | False positives on body outlines |
| XFeat | V2.7 | Regression | Urban-trained, wrong domain |
| ALIKED + LightGlue | V3.2.2+ | Positive for Lynx | aw=0.55 for Lynx in V3.2.2, aw=0.35 in V5.1 |
| KAZE classical | V2.5+ | Positive | Consistent complementary benefit |
| RootSIFT | V2.2+ | Positive (+12.5%) | Free improvement |
| SAM3 mask filtering | V1.3+ | Positive | Background removal for keypoints |
| Salamander yellow/specular mask | V2.8+ | Positive (+1.5%) | Removes false colour keypoints |

## Suggested Next Steps

| Priority | Approach | Expected gain | Complexity | Notes |
|----------|----------|--------------|------------|-------|
| 1 | Pretrained MiewID + pretrained MegaDesc-T-224 (V5.1 configuration) + fresh calibration | Baseline | Low | V5.1 is still the best. Confirm it's reproducible. |
| 2 | BioCLIP 2 as 3rd global backbone | +0.01–0.03 | Medium | Trained on 454k species with 19M image-text pairs. 768-dim, different feature space from MiewID/MegaDesc. Low correlation = potential ensemble value. |
| 3 | DINOv2 ViT-L/14 as 3rd global | +0.01–0.03 | Medium | Strong generalisation backbone, not wildlife-specific. Different failure modes from MiewID. |
| 4 | THL ventral ROI cropping | +0.01–0.03 | Medium | THL is the ONLY uncalibrated species. Biffi et al. 2025 showed HotSpotter 94% on ventral crops. Ventral ROI + KAZE could unlock THL. |
| 5 | Cross-validation calibration | +0.01–0.02 | Medium | K-fold on training identities → more robust weights. Addresses train→test calibration over-fit (V5.4/V5.9 lesson). |
| 6 | Pretrained MiewID + fine-tuned MegaDesc-T-224 fresh calibration | +0.01–0.02 | Low | Combine V5.1's best model with fresh calibration that doesn't give 0 weight to local features. V5.9 dropped ALIKED/SIFT/KAZE entirely for Lynx/Turtle. |
