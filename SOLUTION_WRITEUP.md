# AnimalCLEF 2026 — 9th Place Solution Write-up

**Team:** Sreevaatsav Bavana  
**Final Score:** 0.47075 ARI (private leaderboard)  
**Best Public Score:** 0.51705 ARI (V5.1)  
**Final Rank:** 9th

---

## Abstract

We present a clustering-based pipeline for wildlife individual re-identification that reached 9th place on the AnimalCLEF 2026 private leaderboard. The approach fuses dual global embeddings — a frozen MiewID v3 and a species-specifically fine-tuned MegaDescriptor-T-224 — with a three-component local feature ensemble (SIFT/RootSIFT, KAZE, ALIKED+LightGlue). Clustering uses AgglomerativeClustering with per-species thresholds calibrated via AMI grid-search on training data. The key finding is that fine-tuning MegaDescriptor with ArcFace on CzechLynx (42k images) dramatically improves Lynx embeddings (train AMI +0.42), and that the competition evaluation metric (ARI) is poorly suited for threshold calibration — AMI should be used as the proxy metric instead.

---

## 1. Task Description

AnimalCLEF 2026 is a **clustering** problem: given 2,409 unlabeled test images across four species, group images of the same individual together without any known identity set. This is fundamentally different from the 2025 edition, which was a closed-set classification task.

**Species and test set sizes:**

| Species | Test images | Key challenge |
|---|---|---|
| LynxID2025 | 946 | Camera-trap IR images, no segmentation masks |
| SalamanderID2025 | 689 | Deformable bodies, only ~1.4k training images |
| SeaTurtleID2022 | 500 | High intra-individual variation (angle, lighting) |
| TexasHornedLizards | 274 | **Zero training split** — purely zero-shot |

**Evaluation metric:** Adjusted Rand Index (ARI) — higher is better, maximum 1.0.

**External data permitted:** Any dataset from the WildlifeDatasets package. We used CzechLynx (~42k images) for Lynx fine-tuning.

---

## 2. Final Architecture (V5.1 — best public score)

### 2.1 Overall Pipeline

```
Test images
    │
    ├── Global feature extraction
    │       ├── MiewID v3 (frozen)              → 2152-dim L2-normalized
    │       └── MegaDescriptor-T-224 (fine-tuned) → 768-dim L2-normalized
    │
    ├── Local feature extraction (per-species, filtered by SAM3 mask)
    │       ├── SIFT → RootSIFT + BFMatcher     → match count score
    │       ├── KAZE + BFMatcher                → match count score
    │       └── ALIKED + LightGlue              → match count score
    │
    ├── Weighted ensemble
    │       sim(i,j) = mw·cos(MiewID_i, MiewID_j)
    │                + mgw·cos(Mega_i, Mega_j)
    │                + sw·SIFT_score(i,j)
    │                + kw·KAZE_score(i,j)
    │                + aw·ALIKED_score(i,j)
    │
    └── AgglomerativeClustering (average linkage, precomputed distance)
            distance(i,j) = 1 − sim(i,j)
            threshold calibrated per-species via AMI grid-search
```

### 2.2 Global Models

**MiewID v3** (`conservationxlabs/miewid-msv3`)
- Backbone: EfficientNetV2-RW-M → GeM pooling → BN → 2152-dim output
- Used **frozen** (no fine-tuning)
- TTA: original + horizontal flip → sum → L2-normalize
- Input: 440×440, ImageNet normalization

**MegaDescriptor-T-224** (`BVRA/MegaDescriptor-T-224` via HuggingFace)
- Backbone: Swin-T → 768-dim
- **Fine-tuned per-species** with ArcFace loss (`margin=0.25, scale=16`)
- Optimizer: AdamW (`lr=1e-4, weight_decay=0.01`)
- External data: CzechLynx (~42k images) merged with LynxID2025 training split
- Epochs: Lynx=10, SeaTurtle=10, Salamander=15
- THL: falls back to pretrained MegaDescriptor-L-384 (no training data exists)

Fine-tuned weights uploaded as Kaggle dataset: `sreevaatsavbavana/megadesc-finetuned-v5`

### 2.3 Local Feature Extractors

All local features are computed on the original image, then keypoints are filtered using SAM3 segmentation masks (any keypoint landing on a background pixel is discarded). This avoids the boundary artifact problem of applying masks before extraction.

| Extractor | Descriptor | BFMatcher norm | Score formula | Species used |
|---|---|---|---|---|
| SIFT + RootSIFT | 128-dim float (after sqrt transform) | L2 | `1 - exp(-matches/20)` | All |
| KAZE | 64-dim float (signed, no RootSIFT) | L2 | `1 - exp(-matches/20)` | All |
| ALIKED-n16 + LightGlue | 128-dim (learned) | LightGlue GNN | `1 - exp(-matches/20)` | All |

**RootSIFT transform:**
```python
desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)  # L1 normalize
desc = np.sqrt(desc)                                      # element-wise sqrt
```
Converts L2 to Hellinger distance — more discriminative for histogram-style descriptors.

**Critical bug avoided:** Never apply RootSIFT to KAZE — KAZE produces signed M-SURF descriptors. `sqrt(negative) = NaN` → silent zero-match failures.

**Salamander-specific enhancement:** Yellow-spot ROI mask + specular highlight removal:
```python
# Restrict to yellow spots (H=18-42°, S>80, V>80 in HSV), dilate 25px
# Remove specular highlights (V>220, S<40)
```
Fire salamanders are identified by their yellow spots on black skin. Specular highlights from flash photography account for ~27% of all keypoints but carry zero identity information.

**GPU acceleration:** Top-K=100 candidate preselection via `torch.cdist` before BFMatcher. This reduces O(N²) matching to O(N·K) — mandatory for N=500–946 test images.

### 2.4 Calibrated Ensemble Weights (V5.1)

| Species | mw | mgw | sw | kw | aw | threshold | clusters |
|---|---|---|---|---|---|---|---|
| LynxID2025 | 0.10 | **0.50** | 0.05 | 0.00 | 0.35 | 0.57 | 25/946 |
| SalamanderID2025 | 0.40 | 0.00 | 0.40 | 0.15 | 0.05 | 0.51 | 289/689 |
| SeaTurtleID2022 | 0.50 | 0.20 | 0.15 | 0.10 | 0.05 | 0.63 | 94/500 |
| TexasHornedLizards | 0.275 | 0.275 | 0.15 | 0.10 | 0.20 | 0.30 | 201/274 |

Calibration: AMI grid-search over (mw, mgw, sw, kw) × threshold on training split (≤500 images per species). ALIKED weight `aw = 1 − mw − mgw − sw − kw` (residual). THL has no training split — weights set manually.

### 2.5 SAM3 Segmentation

Segmentation masks from SAM3+YOLO for Salamander, SeaTurtle, and THL (Lynx images are already pre-segmented with black backgrounds by the dataset creators). Masks available from Kaggle dataset: `sreevaatsavbavana/animalclef-26-sam3`.

Masks are used **exclusively** for keypoint filtering — global model inputs are always raw images. Applying masks before global feature extraction was tested and hurt performance.

---

## 3. Experiment History & Ablations

### 3.1 Score Progression

| Version | Score | Δ | Key change |
|---|---|---|---|
| V1 | 0.30655 | — | MiewID v3 + SIFT + BFMatcher baseline |
| V1.2 | 0.26330 | −0.043 | SAM3 white-bg before extraction (FAILED) |
| V1.3 | 0.32528 | +0.019 | SAM3 keypoint mask filtering (correct approach) |
| V2.2 | 0.36607 | +0.041 | RootSIFT transform (2 lines of numpy) |
| V2.3b | 0.36937 | +0.003 | KAZE added to SeaTurtle |
| V2.3c | 0.37726 | +0.008 | KAZE added to THL (larger gain than SeaTurtle) |
| V2.5 | 0.44747 | +0.071 | KAZE all species + AMI threshold calibration |
| V2.6 | 0.45498 | +0.008 | Joint weight+threshold calibration |
| V2.8 | 0.46202 | +0.007 | Salamander yellow-spot ROI mask |
| V3.2.2 | 0.47912 | +0.017 | ALIKED+LightGlue + threshold ceiling fix (0.59→0.75) |
| V4.0 | 0.48600 | +0.007 | Pretrained MegaDescriptor-L-384 as 2nd global model |
| **V5.1** | **0.51705** | **+0.031** | **Fine-tuned MegaDescriptor-T-224 (best public)** |

### 3.2 Key Positive Findings

1. **SAM3 keypoint filtering** (+6.1%): Run extractors on raw images, discard keypoints on background pixels. Never replace background before extraction — SIFT detects strong edges at animal/white boundaries that are pose-dependent artifacts, not identity signal.

2. **RootSIFT** (+12.5% over baseline): Two lines of numpy. Converts L2 distance to Hellinger distance for gradient-histogram descriptors. Free improvement.

3. **KAZE for rigid/textured species** (+7%–9% combined): KAZE's anisotropic diffusion scale space finds stable keypoints where SIFT's Gaussian pyramid blurs dense biological texture (THL spots, turtle scutes, salamander spots). Do NOT apply RootSIFT to KAZE — signed descriptors.

4. **AMI threshold calibration** (+18.6% in V2.5): Grid-search threshold on training set using AMI as proxy metric. This single change produced the largest single-version gain in the competition.

5. **ALIKED + LightGlue for Lynx** (aw=0.35–0.55): Lynx has 0% SAM3 coverage (IR camera-trap images). ALIKED finds meaningful texture patches even without segmentation; LightGlue's GNN matcher filters outliers robustly.

6. **ArcFace fine-tuning for Lynx** (+6.4%): Fine-tuning MegaDescriptor on CzechLynx (42k images) transformed a useless pretrained feature (mgw=0.00 in V4.0) into the dominant signal (mgw=0.50 in V5.1). Train AMI improved from 0.26 to 0.66 (+0.40).

7. **AMI over ARI for calibration**: Even though the competition metric is ARI, using ARI for calibration causes systematic over-merging. ARI rewards large clusters — it pushes thresholds too high and produces catastrophically few clusters (e.g., Lynx collapsed to 8 clusters from 946 images in V4.0-ARI). Always calibrate with AMI.

### 3.3 Key Negative Findings

| What | Why it failed |
|---|---|
| LNBNN scoring (HotSpotter-style) | Background distance ≈ match distance → all LNBNN deltas ≤ 0 → zero local score |
| Alpha Query Expansion on embeddings | MiewID embedding space already optimal — AQE over-smooths, pulls different individuals together |
| XFeat instead of KAZE | XFeat trained on human-made scenes; KAZE gradient-based detectors are better matched to biological textures |
| Isotonic score calibration | Compresses score distributions → thresholds hit ceiling for all species simultaneously |
| 2-phase clustering (known/unknown) | threshold_known=0.30 classified ALL test images as "known" (0 unknowns); cluster naming collision |
| SuperPoint + LightGlue | SP detects corners/junctions — structurally similar across individuals of same species |
| Calibration/test pipeline mismatch | Calibration used ratio-only SIFT scoring; test used RANSAC gate → different score distributions → calibration useless |
| Calibration grid too large (V5.4) | MEGA_W grid expanded to 0.80 → mgw=0.80 overfit to training Lynx distribution → 18 clusters on test |
| Fine-tuned L-384 (V5.2) | LR=3e-5 was 33× too conservative for Swin-L; model barely moved from pretrained weights |
| THL threshold override to 0.40 (V3.2.3) | −27% — "stability elbow" at 0.35-0.40 does NOT indicate optimal threshold for zero-shot species |

### 3.4 What We Couldn't Break

- **Fine-tuned T-224 vs pretrained L-384 for SeaTurtle**: Even with aggressive LLRD fine-tuning, L-384 fine-tuned performed worse than pretrained L-384 + fine-tuned T-224. Model capacity (1536-dim vs 768-dim) matters more than fine-tuning for a species with already-large training data (~8.7k images).

- **Salamander ceiling**: Only ~1.4k training images → insufficient for ArcFace fine-tuning. Fine-tuned T-224 consistently returns mgw=0.00 (calibration ignores it). No large-scale public fire salamander re-ID dataset exists.

- **THL is genuinely zero-shot**: No training split, no comparable external dataset (closest: BalearicLizard, different species, different angle). THL threshold 0.30 was set manually and never improved via calibration.

---

## 4. Public → Private Leaderboard Analysis

**Public best**: 0.51705 (V5.1)  
**Private final**: 0.47075  
**Drop**: −0.047 (−9.1% relative)

Likely causes:

1. **AMI calibration overfit to training distribution**: Thresholds optimized on training identities generalize imperfectly to test identities, which may have different intra/inter-individual similarity distributions (harder images, more challenging viewpoints).

2. **Fine-tuned Salamander model under-powered**: Only 1,388 training images for 587 identities — high identity count with very few examples per identity. ArcFace struggles; embeddings may have memorized training image artifacts rather than learning transferable identity features. Salamander is the hardest species and likely contributes most to the private drop.

3. **THL zero-shot brittleness**: No training data for threshold calibration. The manually set threshold (0.30) was stable on the public LB but may have been more sensitive to the private test distribution.

4. **Possible selected-submission mismatch**: If the final selected submission was not V5.1 (e.g., one of the later versions that regressed on the public LB), the private score would follow the actual submitted notebook, not the public best.

---

## 5. What We Would Do Differently

1. **Fine-tune MegaDescriptor-L-384 correctly** (not L-384 under-trained like V5.2, not T-224 capacity-limited like V5.1). Use LLRD with base LR=1e-3, decay=0.80-0.85, warmup=10%, ArcFace(margin=0.5, scale=64). V5.4 showed embedding-level AMI improvements of +0.33 for Salamander — the ensemble just over-merged on test due to calibration grid over-expansion.

2. **Add more turtle training data**: AmvrakikosTurtles, ReunionTurtles, SouthernProvinceTurtles from the WildlifeDatasets package are all available and would give the fine-tuned model better recall across different turtle populations.

3. **Add GCN-ID (Great Crested Newt) for Salamander**: The only available amphibian re-ID dataset. While different species, it has segmentation masks and a similar morphology problem. Even small additional data might push Salamander fine-tuning above the noise floor.

4. **Better clustering**: Spectral clustering or community detection might outperform agglomerative at identifying the right number of clusters. Agglomerative with average linkage has no explicit model of cluster count — it's entirely threshold-driven.

5. **Never expand calibration grid during a final submission sprint**: The V5.4 lesson — expanding MEGA_W to 0.80 caused catastrophic over-merging. Calibration grids should be designed conservatively and kept fixed across final versions.

---

## 6. Repository Structure

```
AnimalCLEF_26/
├── FINAL_SOLUTION_v5_1/          ← Best public score (0.517)
│   ├── ensemble_global_local_reid_v5_1.ipynb
│   └── build_notebook_v5_1.py
├── FINAL_SOLUTION_v5_12/         ← Latest (two-phase fine-tuning attempt)
│   ├── finetune_global_models_v5_12.ipynb  ← Phase 1+2 fine-tuning
│   └── ensemble_global_local_reid_v5_12.ipynb
├── FINAL_SOLUTION_v4/            ← Architecture reference docs
│   ├── ARCHITECTURE.md
│   └── WINNING_SOLUTIONS_ANALYSIS.md
├── ALL_EXPERIMENTS.md            ← Full experiment log (every version)
├── SOLUTION_COMPARISON.md        ← Side-by-side comparison table
├── FINAL_SOLUTION_v5_8/
│   └── research_notes.md         ← Strategy notes at competition end
├── src/                          ← Source utilities
│   ├── models.py
│   ├── data_loading.py
│   └── inference.py
└── data/raw/                     ← Competition data (metadata.csv + images)
```

### Key Kaggle Datasets

| Dataset slug | Contents |
|---|---|
| `sreevaatsavbavana/megadesc-finetuned-v5` | Fine-tuned MegaDescriptor-T-224 weights (V5.1) |
| `sreevaatsavbavana/animalclef-26-sam3` | SAM3+YOLO segmentation masks |
| `sreevaatsavbavana/version-4-cache` | Precomputed MiewID + MegaDesc embeddings + local features |
| `picekl/czechlynx` | CzechLynx dataset (42k lynx images, 319 IDs) |

---

## 7. References

- AnimalCLEF 2025 overview: https://ceur-ws.org/Vol-4038/paper_231.pdf
- WildFusion (2nd place 2025): https://ceur-ws.org/Vol-4038/paper_253.pdf
- 1st place 2025: https://ceur-ws.org/Vol-4038/paper_245.pdf
- MiewID: https://arxiv.org/abs/2412.05602
- WildlifeReID-10k: https://arxiv.org/abs/2406.09211
- LightGlue: https://github.com/cvg/LightGlue
- WildlifeDatasets package: https://wildlifedatasets.github.io/wildlife-datasets/
