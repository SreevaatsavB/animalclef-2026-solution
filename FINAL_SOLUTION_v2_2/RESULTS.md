# V2.2 Results — RootSIFT + BFMatcher (no LNBNN)

**Score: 0.36607** ← NEW BEST
**Previous best (V1.3): 0.32528**
**Improvement: +0.04079 absolute (+12.5% relative)**

---

## What V2.2 Does Differently from V1.3

One change only: **RootSIFT** applied to SIFT descriptors immediately after
`detectAndCompute`, before the SAM3 keypoint mask filter.

```
Original image → SIFT detectAndCompute → all keypoints + raw descriptors
                                               ↓
                                RootSIFT transform:
                                  desc /= (desc.sum(axis=1) + 1e-7)
                                  desc  = sqrt(desc)
                                               ↓
                              get_seg_mask() filter (V1.3, unchanged)
                                               ↓
                           animal keypoints only → BFMatcher (unchanged)
```

Everything else — MiewID v3 global features, BFMatcher ratio test (0.75),
`1 - exp(-n/20)` score normalization, clustering thresholds — is identical to V1.3.

---

## What This Confirms

**RootSIFT improves BFMatcher discrimination.** The Hellinger metric (achieved
by L1-norm + sqrt before L2 matching) is more discriminative for gradient-histogram
descriptors than raw SIFT + L2. More true-match pairs pass the ratio test; fewer
false-match pairs do.

**LNBNN was the sole structural problem in V2.0/V2.1.** V2.1 kept RootSIFT but
added LNBNN → score collapsed to 0.10957. V2.2 keeps RootSIFT, drops LNBNN →
score rises to 0.36607. LNBNN collapses with a 100k-descriptor background pool;
BFMatcher does not have this problem.

---

## Version History

| Version | Score   | Approach |
|---------|---------|----------|
| V1      | 0.30655 | MiewID v3 + SIFT baseline |
| V1.2    | 0.26330 | + SAM3 white-background — **hurt** |
| V1.3    | 0.32528 | + SAM3 keypoint mask filtering — improved |
| V2.0    | 0.10691 | + MegaDescriptor + RootSIFT + LNBNN — catastrophic |
| V2.1    | 0.10957 | + RootSIFT + LNBNN (diagnostic) — same failure |
| V2.2    | 0.36607 | + RootSIFT only (no LNBNN) — **new best** |

---

## Files

- `build_notebook_v2_2.py` — build script (reads V1.3, applies RootSIFT patch)
- `ensemble_global_local_reid_v2_2.ipynb` — submitted notebook
- `kernel-metadata.json` — Kaggle metadata

---

## Key Lesson

RootSIFT (L1-norm + sqrt) is a free +4% improvement over raw SIFT when using
BFMatcher with L2 distance. The transform converts Euclidean distance to
Hellinger distance, which is more appropriate for gradient-histogram descriptors.

LNBNN should only be used when the background descriptor pool is small (K × kpts ≪ 10k).
With K=100 and ~1000 keypoints per image, the pool is ~100k descriptors — the
background distance collapses to near-zero and LNBNN scores become zero for most
pairs. BFMatcher + ratio test does not have this structural failure mode.
