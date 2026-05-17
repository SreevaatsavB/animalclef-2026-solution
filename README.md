# AnimalCLEF 2026 — 9th Place Solution

Wildlife individual re-identification as clustering.
**Final score: 0.47075 ARI | Rank: 9th | Best public: 0.51705 ARI**

---

## Competition

[AnimalCLEF 2026](https://www.kaggle.com/competitions/animal-clef-2026) is a wildlife clustering task:
given 2,409 unlabeled test images across four species, group images of the same individual without any known identity set. Evaluated by **Adjusted Rand Index (ARI)**.

| Species | Test images | Challenge |
|---|---|---|
| LynxID2025 | 946 | IR camera-trap, no segmentation |
| SalamanderID2025 | 689 | Deformable body, only 1.4k train images |
| SeaTurtleID2022 | 500 | High intra-individual variation |
| TexasHornedLizards | 274 | Zero training split — fully zero-shot |

---

## Architecture

```
Test images
    │
    ├── Global features
    │       ├── MiewID v3 (frozen, 2152-dim)
    │       └── MegaDescriptor-T-224 (ArcFace fine-tuned per-species, 768-dim)
    │
    ├── Local features (SAM3 mask-filtered keypoints)
    │       ├── SIFT + RootSIFT → BFMatcher
    │       ├── KAZE → BFMatcher
    │       └── ALIKED + LightGlue
    │
    └── Weighted ensemble → AgglomerativeClustering
            (AMI-calibrated thresholds per species)
```

**Best calibrated weights (V5.1):**

| Species | MiewID | MegaDesc | SIFT | KAZE | ALIKED | Threshold |
|---|---|---|---|---|---|---|
| Lynx | 0.10 | **0.50** | 0.05 | 0.00 | 0.35 | 0.57 |
| Salamander | 0.40 | 0.00 | 0.40 | 0.15 | 0.05 | 0.51 |
| SeaTurtle | 0.50 | 0.20 | 0.15 | 0.10 | 0.05 | 0.63 |
| THL | 0.275 | 0.275 | 0.15 | 0.10 | 0.20 | 0.30 |

---

## Score History

| Version | Score | Key change |
|---|---|---|
| V1 | 0.307 | MiewID + SIFT baseline |
| V1.3 | 0.325 | SAM3 keypoint mask filtering |
| V2.2 | 0.366 | RootSIFT (2 lines of numpy, +12.5%) |
| V2.3c | 0.377 | KAZE for THL and SeaTurtle |
| V2.5 | 0.447 | KAZE all species + AMI threshold calibration (+18.6%) |
| V2.8 | 0.462 | Salamander yellow-spot ROI mask |
| V3.2.2 | 0.479 | ALIKED+LightGlue + threshold ceiling fix |
| V4.0 | 0.486 | MegaDescriptor-L-384 as 2nd global model |
| **V5.1** | **0.517** | **ArcFace fine-tuned MegaDescriptor (CzechLynx data)** |

---

## Key Findings

1. **SAM3 keypoint filtering**: Run extractors on raw images, then discard keypoints on background pixels. Never replace background before extraction — SIFT detects boundary edges as identity-irrelevant keypoints.

2. **RootSIFT**: Two lines of numpy (`L1 normalize → sqrt`). Converts L2 to Hellinger distance for SIFT descriptors. Free +12.5% improvement.

3. **Calibrate with AMI, not ARI**: The competition metric is ARI, but using ARI for threshold calibration causes systematic over-merging (rewards large clusters). AMI-calibrated thresholds consistently outperform ARI-calibrated ones.

4. **ArcFace fine-tuning + external data for Lynx**: CzechLynx (42k images) is critical. Train AMI went from 0.26 → 0.66 after fine-tuning MegaDescriptor on this dataset. Without external data, MegaDescriptor contributes nothing for Lynx.

5. **Salamander is the ceiling**: Only 1.4k training images for 587 identities. Fine-tuning never helps. No large-scale public fire salamander re-ID dataset exists.

6. **Never apply RootSIFT to KAZE**: KAZE produces signed M-SURF descriptors. `sqrt(negative) = NaN` → silent zero-match failure.

---

## Repo Structure

```
AnimalCLEF_26/
├── FINAL_SOLUTION_v5_1/          ← Best public score (0.517)
│   ├── ensemble_global_local_reid_v5_1.ipynb
│   └── build_notebook_v5_1.py
├── FINAL_SOLUTION_v5_12/         ← Latest (two-phase resumable fine-tuning)
│   ├── finetune_global_models_v5_12.ipynb
│   └── ensemble_global_local_reid_v5_12.ipynb
├── FINAL_SOLUTION_v4/            ← Architecture reference & research docs
├── ALL_EXPERIMENTS.md            ← Full log: every version, scores, root causes
├── SOLUTION_COMPARISON.md        ← Side-by-side comparison table
├── SOLUTION_WRITEUP.md           ← Full technical writeup (paper-style)
├── src/                          ← Utility modules
│   ├── models.py
│   ├── data_loading.py
│   └── inference.py
└── data/raw/                     ← Competition data (not tracked in git)
```

---

## Setup

```bash
# Activate environment
source venv_animalclef2026/bin/activate

# Verify
python -c "import wildlife_tools, torch; print(torch.__version__)"
```

Key packages: `wildlife-tools`, `wildlife-datasets`, `timm`, `torch`, `kornia`, `lightglue`

---

## Running on Kaggle

The submission notebooks are designed to run in Kaggle kernels. Required input datasets:

| Kaggle slug | Contents |
|---|---|
| `animal-clef-2026` | Competition data |
| `sreevaatsavbavana/megadesc-finetuned-v5` | Fine-tuned MegaDescriptor weights |
| `sreevaatsavbavana/animalclef-26-sam3` | SAM3 segmentation masks |
| `sreevaatsavbavana/version-4-cache` | Precomputed embeddings + local features |
| `picekl/czechlynx` | CzechLynx (42k lynx images, for fine-tuning) |

To reproduce V5.1 (best): run `FINAL_SOLUTION_v5_1/ensemble_global_local_reid_v5_1.ipynb` with the above datasets attached.

---

## Full Write-up

See [`SOLUTION_WRITEUP.md`](./SOLUTION_WRITEUP.md) for the complete technical write-up including all ablations, root-cause analyses for every regression, and a discussion of the public→private leaderboard drop.

---

## References

- [MiewID](https://arxiv.org/abs/2412.05602) — `conservationxlabs/miewid-msv3`
- [MegaDescriptor / WildlifeReID-10k](https://arxiv.org/abs/2406.09211) — `BVRA/MegaDescriptor-*`
- [LightGlue](https://github.com/cvg/LightGlue) — attentional GNN keypoint matcher
- [WildlifeDatasets](https://wildlifedatasets.github.io/wildlife-datasets/) — unified wildlife re-ID datasets
- [AnimalCLEF 2025 overview](https://ceur-ws.org/Vol-4038/paper_231.pdf)
