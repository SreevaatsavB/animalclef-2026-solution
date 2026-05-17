# V2.3c Results — KAZE for TexasHornedLizards

**Score: 0.37726** ← NEW BEST
**Previous best (V2.3b): 0.36937**
**Improvement: +0.00789 absolute (+2.1% relative)**

---

## What V2.3c Does Differently from V2.3b

Added `KAZEExtractor` for TexasHornedLizards. SeaTurtleID2022 unchanged.

| Species | V2.3b | V2.3c |
|---------|-------|-------|
| SalamanderID2025 | global=0.70, sift=0.30 | unchanged |
| SeaTurtleID2022 | global=0.65, sift=0.20, kaze=0.15 | unchanged |
| LynxID2025 | global=0.70, sift=0.30 | unchanged |
| TexasHornedLizards | global=0.65, sift=0.35 | global=0.65, sift=0.20, **kaze=0.15** |

---

## Rationale

TexasHornedLizards shares the same properties that made KAZE beneficial for SeaTurtle:
- **Rigid body** → keypoints are stable across views and poses
- **100% SAM3 cache coverage** → every image benefits from mask-filtered keypoints
- **Dense surface patterns** (spots/scales) → non-linear scale space finds different,
  stable keypoints compared to SIFT's Gaussian pyramid

No new code needed — `KAZEExtractor` and `get_extractor('kaze')` routing already
exist from V2.3b. Only the SPECIES_CONFIG patch is applied.

---

## KAZEExtractor (inherited from V2.3b)

- `cv2.KAZE_create(upright=False)` — rotation invariant, pure OpenCV
- Max 1000 keypoints by response strength
- No RootSIFT (M-SURF signed descriptors; sqrt of negative → NaN)
- SAM3 mask filter applied (100% coverage for both SeaTurtle and TexasLizard)
- 64-dim float32 descriptors, matched via `batch_sift_match_gpu` (torch.cdist L2)

---

## Version History

| Version | Score   | Approach |
|---------|---------|----------|
| V1.3    | 0.32528 | Raw SIFT + SAM3 mask |
| V2.2    | 0.36607 | + RootSIFT |
| V2.3a   | 0.36378 | + Uniform 0.65/0.35 weights — regression |
| V2.3b   | 0.36937 | + KAZE for SeaTurtle — **best** |
| V2.3c   | 0.37726 | + KAZE for TexasHornedLizards — **new best** |

---

## Key Lesson

KAZE gave a larger gain for TexasHornedLizards (+0.00789) than it did for SeaTurtle (+0.00330).
This confirms the user's intuition: **local features carry more discriminative signal for THL**.
Likely reasons:
- THL has very dense, high-frequency spot/scale patterns → KAZE's non-linear scale space finds
  stable keypoints in these regions where SIFT's Gaussian pyramid may blur them
- Each individual has a unique spot pattern — more inter-individual variation in local texture
  than SeaTurtle (which has shell plate geometry as the main discriminator)
- 100% SAM3 coverage means every THL image benefits from mask-filtered keypoints

The asymmetry between species (+2.1% for THL vs +0.9% for SeaTurtle from KAZE) suggests
that for future experiments, THL is the species most likely to benefit from additional
local feature improvements.

---

## Files

- `build_notebook_v2_3c.py` — build script
- `ensemble_global_local_reid_v2_3c.ipynb` — submitted notebook
- `kernel-metadata.json` — Kaggle metadata
