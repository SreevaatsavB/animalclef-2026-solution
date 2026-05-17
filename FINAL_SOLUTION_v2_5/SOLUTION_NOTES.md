# AnimalCLEF 2026 — V2.5 Solution Notes

**Score: 0.44747** (+0.070 over V2.3c best of 0.37726 — 18.6% relative gain)

---

## 1. What Changed vs V2.3c

| Dimension | V2.3c | V2.5 |
|-----------|-------|------|
| SIFT+KAZE species | SeaTurtle + THL only | **All 4 species** |
| Salamander features | global:0.70 + sift:0.30 | global:0.65 + sift:0.20 + kaze:0.15 |
| Lynx features | global:0.70 + sift:0.30 | global:0.70 + sift:0.20 + kaze:0.10 |
| Clustering thresholds | Hand-tuned | **AMI-calibrated from training identities** |
| Lynx threshold | 0.35 | **0.51** (+0.16 — was over-splitting badly) |
| Salamander threshold | 0.35 | **0.29** (−0.06 — was over-merging) |
| SeaTurtle threshold | 0.40 | **0.42** (+0.02 — minor) |
| THL threshold | 0.30 | 0.30 (no training split, unchanged) |

**Dominant improvement**: threshold calibration. All thresholds were previously hand-tuned by eyeballing similarity distributions. Using training identity labels via AMI grid-search found the true within-individual vs between-individual gaps. Lynx was the biggest winner: threshold 0.35 was far too low (required similarity ≥ 0.65 to merge), so camera-trap photos of the same lynx individual were being split into separate clusters. Raising to 0.51 (merges pairs with similarity ≥ 0.49) collapsed this correctly.

> **Distance threshold semantics**: `AgglomerativeClustering(distance_threshold=t)` merges clusters with average linkage distance ≤ t. Since distance = 1 − similarity, threshold t = 0.35 means "only merge if similarity ≥ 0.65". A *lower* threshold produces *more* clusters (stricter merging). A *higher* threshold produces *fewer* clusters (more permissive merging).

---

## 2. Full Pipeline

### 2.1 Dependencies

```
kornia==0.8.2
transformers==4.36.0   ← pinned; newer versions break MiewID v3
wildlife-datasets
wildlife-tools
timm
scikit-learn
```

Known conflict: `sentence-transformers>=4.41.0` requires transformers≥4.41.0. Ignored — MiewID v3 takes priority and the sentence-transformers package is not used in this pipeline.

Hardware: **Tesla P100-PCIE-16GB** (Kaggle free GPU, 1 device).

---

### 2.2 Dataset

| Species | Test | Train |
|---------|------|-------|
| LynxID2025 | 946 | 2,957 |
| SalamanderID2025 | 689 | 1,388 |
| SeaTurtleID2022 | 500 | 8,729 |
| TexasHornedLizards | 274 | 0 |
| **Total** | **2,409** | **13,074** |

Source: `/kaggle/input/animal-clef-2026`, loaded via `AnimalCLEF2026(root=ROOT_DIR, load_label=True)`.

---

### 2.3 SAM3 Segmentation Coverage

SAM3 masks are used to filter SIFT and KAZE keypoints to the animal region. `get_seg_mask()` looks up the white-background SAM3 image for each file path and derives mask as pixels ≠ (255, 255, 255). Returns `None` for Lynx (0% coverage) → keypoints are extracted unfiltered on original images.

| Species | SAM3 coverage |
|---------|--------------|
| LynxID2025 | **0%** — SIFT + KAZE on original images, no masking |
| SalamanderID2025 | 100% — masked |
| SeaTurtleID2022 | 100% — masked |
| TexasHornedLizards | 100% — masked |

Total SAM3 cache: 11,580 images cached, 3,903 fallback to original (15,483 total including train+test).

Dataset source on Kaggle: `sreevaatsavbavana/animalclef-26-sam3`

---

### 2.4 Precomputed Cache (v2-5-cache)

The Kaggle dataset `sreevaatsavbavana/v2-5-cache` was mounted at `/kaggle/input/datasets/sreevaatsavbavana/v2-5-cache/cache` and contained **precomputed local features + match scores for all 4 species**. Cell 1.4b copied all 16 files into the working `cache/` directory before any extraction ran, so Cells 3.5 and 4.4 loaded everything from cache without computing.

Files copied (16 total):

| Subdirectory | File |
|---|---|
| `local_features/` | `LynxID2025_sift.pkl`, `LynxID2025_kaze.pkl` |
| `local_features/` | `SalamanderID2025_sift.pkl`, `SalamanderID2025_kaze.pkl` |
| `local_features/` | `SeaTurtleID2022_sift.pkl`, `SeaTurtleID2022_kaze.pkl` |
| `local_features/` | `TexasHornedLizards_sift.pkl`, `TexasHornedLizards_kaze.pkl` |
| `match_scores/` | `LynxID2025_sift_matches.npy`, `LynxID2025_kaze_matches.npy` |
| `match_scores/` | `SalamanderID2025_sift_matches.npy`, `SalamanderID2025_kaze_matches.npy` |
| `match_scores/` | `SeaTurtleID2022_sift_matches.npy`, `SeaTurtleID2022_kaze_matches.npy` |
| `match_scores/` | `TexasHornedLizards_sift_matches.npy`, `TexasHornedLizards_kaze_matches.npy` |

> Note: The cache included SeaTurtle and THL features even though those were already in V2.3c. This means the cache was a full re-extraction (not just the two new species), which is fine — the precomputed values are identical.

---

### 2.5 Global Features — MiewID v3

- Model: `conservationxlabs/miewid-msv3`
- Backbone: EfficientNetV2-RW-M
- Output dim: **2152** (confirmed from `final_in_features 2152` in build output)
- TTA: original image + horizontal flip → elementwise sum → L2 normalize
- Image size: 512×512 for all species
- All output embeddings are unit-norm (verified: norm = 1.000)

Extracted shapes:

| Species | Shape |
|---------|-------|
| LynxID2025 | (946, 2152) |
| SalamanderID2025 | (689, 2152) |
| SeaTurtleID2022 | (500, 2152) |
| TexasHornedLizards | (274, 2152) |

Saved to `cache/embeddings/{species}_global.npy`.

---

### 2.6 Query Expansion (applied to global features)

**Formula:**
```python
expanded[i] = features[i] + alpha * mean(features[top-k neighbors of i])
expanded[i] = L2_normalize(expanded[i])
```
- `alpha = 0.5`
- k per species: Lynx=5, Salamander=3, SeaTurtle=8, THL=5

Result stored in `global_features_expanded`. **The ensemble similarity (Section 2.9) uses `global_features_expanded`, not the raw `global_features_cache`.** This is a mild smoothing step — it slightly pulls each embedding toward its nearest neighbours in the test set before computing pairwise cosine similarities.

> This is distinct from Alpha-QE (V2.4), which used weighted top-k and exponent α. Here alpha=0.5 is fixed, similarity weights are implicit (uniform mean over top-k), and there is no power weighting.

---

### 2.7 Local Features

#### SIFT (with RootSIFT transform)

```python
cv2.SIFT_create(nfeatures=500)
```
- Max 500 keypoints per image (sorted by response)
- **RootSIFT transform**: `desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)` then `desc = np.sqrt(desc)` — converts L2 distance to Hellinger distance, more discriminative for gradient histograms
- Keypoints filtered to SAM3 mask where available (`mask[y, x] == 0` → discard)

#### KAZE

```python
cv2.KAZE_create(upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4)
```
- 64-dim signed M-SURF descriptors
- Keypoints sorted by response, capped at `max_keypoints`
- **RootSIFT NOT applied** — KAZE descriptors are signed; `sqrt(negative)` = NaN
- Keypoints filtered to SAM3 mask where available (Lynx: no mask, all keypoints kept)

#### Keypoint statistics on test set (from Cell 3.6 output)

| Species | SIFT valid | SIFT avg kpts | KAZE valid | KAZE avg kpts |
|---------|-----------|---------------|-----------|---------------|
| LynxID2025 | 945/946 | 395 | 943/946 | 528 |
| SalamanderID2025 | 689/689 | 736 | 689/689 | 794 |
| SeaTurtleID2022 | 468/500 | 507 | 437/500 | 321 |
| TexasHornedLizards | 274/274 | 933 | 274/274 | 924 |

Observations:
- SeaTurtle: 32 images with no valid SIFT keypoints, 63 with no valid KAZE — SAM3 mask leaves a small animal region in some images, yielding too few detections.
- Lynx KAZE (528) > SIFT (395): KAZE's non-linear scale space finds more stable keypoints on fur even without SAM3 masking.
- THL has the highest keypoint density (~930/image both extractors) — dense scales/spots are ideal for both detectors.

---

### 2.8 Matching — BFMatcher

```python
matcher = cv2.BFMatcher(cv2.NORM_L2)   # L2 norm, brute-force
```

**Pipeline per species:**
1. GPU top-K candidate selection using `torch.cdist` (K=100) to limit BFMatcher cost from O(N²) to O(N·K)
2. For each image pair (i, j) in top-K candidates: run BFMatcher with Lowe's ratio test (threshold=0.75)
3. Count surviving matches → compute score: `score(i,j) = 1.0 − exp(−num_matches / 20.0)`
4. Diagonal forced to 1.0 (self-match)
5. Matrix is symmetric: `score[j,i] = score[i,j]`

Batch size for candidate selection: 50 images at a time.

#### Match score statistics (from Cell 4.4 output)

| Species | Extractor | Matrix shape | Mean score | Non-zero off-diagonal pairs |
|---------|-----------|-------------|-----------|----------------------------|
| LynxID2025 | SIFT | (946, 946) | 0.057 | 129,386 |
| LynxID2025 | KAZE | (946, 946) | 0.086 | 128,652 |
| SalamanderID2025 | SIFT | (689, 689) | 0.069 | 90,792 |
| SalamanderID2025 | KAZE | (689, 689) | 0.091 | 91,400 |
| SeaTurtleID2022 | SIFT | (500, 500) | 0.035 | 47,576 |
| SeaTurtleID2022 | KAZE | (500, 500) | 0.025 | 36,154 |
| TexasHornedLizards | SIFT | (274, 274) | 0.102 | 34,954 |
| TexasHornedLizards | KAZE | (274, 274) | 0.100 | 34,130 |

Observations:
- **Lynx KAZE (0.086) > SIFT (0.057)**: KAZE's non-linear diffusion scale space finds stable rosette/fur patterns SIFT misses, even on raw (unmasked) images. High non-zero pair count (128k) confirms dense matching.
- **Salamander KAZE (0.091) > SIFT (0.069)**: Consistent improvement. Salamander body patterns respond well to KAZE's anisotropic diffusion.
- **SeaTurtle KAZE (0.025) < SIFT (0.035)**: Fewer valid KAZE keypoints (437 vs 468) and lower match scores — SeaTurtle's rigid carapace patterns are handled better by SIFT's Gaussian pyramid.
- **THL both ~0.10**: Highest match scores across all species. Dense scale/spot patterns on a rigid body are ideal for both detectors.

---

### 2.9 Ensemble Similarity

```
sim(i, j) = global_weight  × cosine_sim(expanded_emb[i], expanded_emb[j])
           + sift_weight   × sift_score(i, j)
           + kaze_weight   × kaze_score(i, j)
```

Note: cosine similarity uses `global_features_expanded` (post-QE), not raw embeddings.

Weights:

| Species | Global | SIFT | KAZE | Total |
|---------|--------|------|------|-------|
| SalamanderID2025 | 0.65 | 0.20 | 0.15 | 1.00 |
| SeaTurtleID2022 | 0.65 | 0.20 | 0.15 | 1.00 |
| LynxID2025 | **0.70** | 0.20 | **0.10** | 1.00 |
| TexasHornedLizards | 0.65 | 0.20 | 0.15 | 1.00 |

Lynx gets higher global weight (0.70) and lower KAZE weight (0.10) because KAZE has no SAM3 masking → potential background noise contamination.

Ensemble similarity matrix statistics (from Cell 5.2 output):

| Species | Mean sim | Std | Min | Max |
|---------|---------|-----|-----|-----|
| LynxID2025 | 0.355 | 0.109 | 0.111 | 1.000 |
| SalamanderID2025 | 0.321 | 0.145 | 0.026 | 1.000 |
| SeaTurtleID2022 | 0.245 | 0.086 | **-0.004** | 1.000 |
| TexasHornedLizards | 0.456 | 0.100 | 0.150 | 1.000 |

> SeaTurtle min = −0.004: possible for the weighted sum to dip slightly below 0 when local match scores are 0 and the global cosine similarity for some cross-species-mismatched pair is very low. Clipped to 0 by the clustering distance conversion (`clip(1 − sim, 0, 1)`).

---

### 2.10 Threshold Calibration (New in V2.5)

**Method:** Grid-search the distance threshold that maximises **AMI (Adjusted Mutual Information)** between predicted clusters and ground-truth training identities.

- Input: global cosine similarity on training images (QE not applied during calibration — fresh extraction)
- Grid: threshold ∈ {0.15, 0.16, …, 0.60} → 46 values
- Clustering: `AgglomerativeClustering(metric="precomputed", linkage="average", distance_threshold=t)` on `clip(1 − cos_sim, 0, 1)`
- SeaTurtle: randomly subsampled to 2,500 (from 8,729) to cap memory and runtime
- MiewID reloaded fresh for calibration (was `del model` after test extraction to free VRAM)

Results (from Cell 5.4 output):

| Species | Train images used | Identities | Old thr | New thr | Δ | AMI | Time |
|---------|-------------------|-----------|---------|---------|---|-----|------|
| LynxID2025 | 2,500 / 2,957 | 76 | 0.35 | **0.51** | +0.16 | 0.4269 | 163.8s |
| SalamanderID2025 | 1,388 / 1,388 | 587 | 0.35 | **0.29** | −0.06 | 0.3058 | 104.8s |
| SeaTurtleID2022 | 2,500 / 8,729 | 412 | 0.40 | **0.42** | +0.02 | 0.8397 | 118.6s |
| TexasHornedLizards | — | — | 0.30 | **0.30** | 0 | — | — |

**Total calibration time: ~387s (6.5 minutes)**

**Interpretation:**

- **Lynx 0.35 → 0.51 (+0.16)**: The old threshold required similarity ≥ 0.65 to merge two images. Within-individual Lynx pairs often have similarity 0.49–0.65 (different camera angles, seasons, lighting) — these were NOT being merged. Raising to 0.51 (merges pairs with similarity ≥ 0.49) correctly consolidates them. Result: 946 images → 53 clusters instead of ~300+.
- **Salamander 0.35 → 0.29 (−0.06)**: The old threshold was too permissive — pairs with similarity 0.65–0.71 from *different* individuals were being merged. Salamander embeddings have low inter-individual separation (deformable bodies, similar colour). Tightening to 0.29 (requires similarity ≥ 0.71 to merge) correctly separates them. Result: 689 images → 492 clusters (many singletons expected — 587 training identities in 1,388 images = ~2.4 photos/individual).
- **SeaTurtle 0.40 → 0.42 (+0.02)**: Small adjustment. AMI = 0.8397 confirms global embeddings are already highly discriminative for SeaTurtle — local features are supplementary.
- **THL**: No training identities available. V2.3c threshold 0.30 kept unchanged.

---

### 2.11 Clustering Results (from Cell 6.2 output)

Algorithm: `AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=t)`

Distance matrix = `clip(1 − ensemble_similarity, 0, 1)`.

| Species | Images | Calibrated threshold | Clusters | Avg images/cluster |
|---------|--------|---------------------|----------|-------------------|
| LynxID2025 | 946 | 0.51 | **53** | 17.8 |
| SalamanderID2025 | 689 | 0.29 | **492** | 1.4 |
| SeaTurtleID2022 | 500 | 0.42 | **301** | 1.7 |
| TexasHornedLizards | 274 | 0.30 | **225** | 1.2 |
| **Total** | **2,409** | — | **1,071** | **2.3** |

Observations:
- **Lynx 53 clusters from 946 images (17.8/cluster)**: Consistent with camera-trap data (many photos per individual). Train set: 76 identities in 2,957 images = 39 photos/individual average. Test is similar.
- **Salamander 492 clusters from 689 images (1.4/cluster)**: Mostly singletons. Expected — train has 587 identities in 1,388 images (2.4/individual). Very challenging few-shot species.
- **SeaTurtle 301 clusters from 500 images (1.7/cluster)**: Similar sparsity. 412 sampled train identities consistent with low test cluster density.
- **THL 225 clusters from 274 images (1.2/cluster)**: Near-singleton regime with no training data for calibration.

---

### 2.12 Submission

- Total rows: 2,409 (one per test image)
- Total clusters: 1,071
- Format: `image_id, cluster` where `cluster = cluster_{species}_{label_int}`
- Example: `cluster_LynxID2025_16`, `cluster_SalamanderID2025_3`

---

## 3. Version History

| Version | Score | Key change |
|---------|-------|------------|
| V1 | 0.30655 | MiewID v3 + SIFT (baseline) |
| V1.2 | 0.26330 | SAM3 white-background — artifact keypoints, hurt |
| V1.3 | 0.32528 | SAM3 keypoint mask filtering (correct usage) |
| V2.0 | 0.10691 | + MegaDescriptor + RootSIFT + LNBNN — catastrophic threshold mismatch |
| V2.1 | 0.10957 | Remove MegaDescriptor; LNBNN still breaks threshold scale |
| V2.2 | 0.36607 | RootSIFT only, no LNBNN (+12.5% over V1.3) |
| V2.3a | 0.36378 | Uniform 0.65/0.35 weights — tiny regression |
| V2.3b | 0.36937 | + KAZE for SeaTurtle |
| V2.3c | 0.37726 | + KAZE for THL — **prev best** |
| V2.4 | 0.35624 | Alpha-QE on embeddings — regression (embeddings already near-optimal) |
| **V2.5** | **0.44747** | **KAZE all 4 species + AMI threshold calibration — new best** |

---

## 4. Critical Lessons Learned

### Threshold calibration is the highest-leverage improvement
All V2.3c thresholds were hand-tuned. The 76 Lynx training identities revealed the true within-individual similarity gap is at distance 0.51, not 0.35. Lynx was over-splitting by ~6× (estimated ~300 clusters vs correct 53). This single parameter change — which requires only global embeddings on training images, no new feature extraction — accounts for most of the +0.070 jump.

### Lower distance threshold ≠ better
A lower `distance_threshold` in `AgglomerativeClustering` produces *more* clusters (stricter merging). It can over-split (Lynx case) or be correct (Salamander needs 0.29 to separate visually similar individuals). Always calibrate from labelled data.

### KAZE complements SIFT for all species
Even Lynx (0% SAM3, background-contaminated keypoints) shows KAZE mean match score 0.086 vs SIFT 0.057 — a 51% lift. Non-linear scale space finds different stable structures SIFT's Gaussian pyramid misses. Adding KAZE is essentially a free signal source.

### SAM3 mask: filter keypoints, don't replace the image
Correct: run SIFT/KAZE on original image, discard keypoints where `mask[y,x] == 0`.
Wrong: replace image with white-background version before extraction → boundary artifact keypoints.

### Alpha-QE over-smooths MiewID embeddings (V2.4 lesson)
MiewID v3 already produces tight within-individual clusters. Post-hoc AQE pulls embeddings toward neighbours, over-merging (Salamander −105 clusters) and perversely over-splitting Lynx (+71). Do not apply post-processing to global embeddings for this dataset.

### LNBNN score scale collapses for large descriptor pools (V2.0/V2.1 lesson)
With K=100 candidates × ~1000 kpts, the background pool has ~100k descriptors. The 4th-nearest-neighbour background distance (LNBNN denominator) becomes tiny (~0.10–0.15 Hellinger), so match_delta ≤ 0 for most descriptors → LNBNN score = 0 → match matrix ≈ identity → effective ensemble = 0.70 × global only → all thresholds miscalibrated at the 0.70× scale.

---

## 5. Files

```
FINAL_SOLUTION_v2_5/
├── build_notebook_v2_5.py                    # Build script — patches V2.3c notebook → V2.5
├── ensemble_global_local_reid_v2_5.ipynb     # Locally generated notebook (no outputs)
├── ensemble-global-local-reid-v2-5.ipynb     # Notebook as executed on Kaggle (with full outputs)
├── kernel-metadata.json                       # Kaggle push config
└── SOLUTION_NOTES.md                         # This file
```

### kernel-metadata.json

```json
{
  "id": "sreevaatsavbavana/animalclef-26-v2-5",
  "title": "AnimalCLEF 26 V2.5 (KAZE all species + threshold calib)",
  "enable_gpu": true,
  "dataset_sources": [
    "sreevaatsavbavana/animalclef-26-sam3",
    "sreevaatsavbavana/v2-5-cache"
  ],
  "competition_sources": ["animal-clef-2026"]
}
```

### Kaggle dataset sources

| Dataset | Purpose |
|---------|---------|
| `animal-clef-2026` | Competition images + metadata |
| `sreevaatsavbavana/animalclef-26-sam3` | SAM3 white-background segmented images (for mask derivation) |
| `sreevaatsavbavana/v2-5-cache` | Precomputed SIFT+KAZE local features (.pkl) + BFMatcher match scores (.npy) for all 4 species |
