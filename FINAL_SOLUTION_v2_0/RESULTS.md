# V2.0 Results — MegaDescriptor + RootSIFT + LNBNN

**Score: 0.10691**
**Previous best (V1.3): 0.32528**
**Regression: -0.21837 absolute (-67% relative) — catastrophic failure**

---

## What V2.0 Added (all at once)

Three simultaneous changes on top of V1.3:

1. **MegaDescriptor-L-384 fusion** — Swin-L global embeddings concatenated with MiewID v3 → 3688-dim fused embedding (42% MegaDescriptor + 58% MiewID, L2-renormalized)
2. **RootSIFT** — L1-norm + sqrt applied to SIFT descriptors before BFMatcher/LNBNN matching
3. **LNBNN scoring** — HotSpotter-style local Naive Bayes Nearest Neighbor scoring replacing BFMatcher ratio test

---

## Bugs Found and Fixed (before submission)

These implementation bugs were caught during code review. They did NOT cause the score drop — the LNBNN structural issue is the root cause.

| Bug | Cell | Symptom | Fix |
|-----|------|---------|-----|
| Missing `num_classes=0` in `timm.create_model` | Cell 2.x | Model returns class logits (shape mismatch), not embeddings | Added `num_classes=0` argument |
| Self-image in LNBNN `db_descs_list` | Cell 5.x | Query image included in its own background pool → background distance distorted downward | Pass `None` for `j==i` in db_descs_list |
| `np.add.at` accumulation | Cell 5.x | Correct but very slow Python loop | Replaced with GPU `scatter_add_` |
| `np.argsort` for top-K | Cell 5.x | O(n log n) sort for top-100 | Replaced with `np.argpartition` (O(n)) |
| Python loop → vectorised | Cell 5.x | Slow score accumulation | Replaced with `np.maximum` broadcasting |

---

## Failure Analysis

### Why 0.10 instead of 0.30?

Score ~0.10 is not a marginal regression — it implies every image was assigned to its own individual (every singleton cluster). This is structural failure in the clustering step.

### LNBNN Near-Zero Scores

With K=100 candidates × ~1000 keypoints per image = ~100,000 descriptors in the LNBNN pool:

- The background distance (k+1 = 4th nearest neighbor among 100k similar-animal descriptors) is very small: ~0.10–0.15 in Hellinger space
- For most query descriptors: `bg_dist < match_dist` → `delta ≤ 0` → LNBNN score = 0
- Most image pairs accumulate zero LNBNN score → `match_matrix ≈ identity`

### Ensemble Scale Collapse

```
V1.3 (working):  ensemble = 0.70 * global_sim + 0.30 * bfmatcher_score (~0.20 avg)
V2.0 (broken):   ensemble = 0.70 * global_sim + 0.30 * 0              (LNBNN = 0)
```

The effective ensemble is `0.70 * global_sim` — values are ~0.30 lower than what the clustering thresholds were calibrated for.

### Threshold Miscalibration

Clustering distance thresholds were tuned for V1.3's full ensemble scale:

```
Same-individual pair (global_sim ≈ 0.8):
  V1.3 distance = 1 - (0.70*0.8 + 0.30*0.20) = 1 - 0.62 = 0.38 → within threshold 0.35? Maybe
  V2.0 distance = 1 - (0.70*0.8 + 0)         = 1 - 0.56 = 0.44 → > threshold 0.35 → NOT clustered
```

With LNBNN scores collapsing to zero, the distance between same-individual images exceeds the clustering threshold → every image becomes its own cluster → score ~0.10.

### MegaDescriptor: NOT the culprit

Hypothesis at the time: MegaDescriptor might be poisoning global similarity. Disproved by V2.1, which scored 0.10957 with identical LNBNN behavior and no MegaDescriptor. See `FINAL_SOLUTION_v2_1/RESULTS.md`.

---

## Version History

| Version | Score   | Approach |
|---------|---------|----------|
| V1      | 0.30655 | MiewID v3 + SIFT (baseline) |
| V1.2    | 0.26330 | + SAM3 white-background replacement — **hurt** |
| V1.3    | 0.32528 | + SAM3 keypoint mask filtering — **best** |
| V2.0    | 0.10691 | + MegaDescriptor + RootSIFT + LNBNN — **catastrophic** |

---

## Files

- `build_notebook_v2_0.py` — build script (reads V1.3, patches, writes V2.0)
- `ensemble_global_local_reid_v2_0.ipynb` — submitted notebook
- `kernel-metadata.json` — Kaggle metadata

---

## Key Lessons

1. **Never add multiple experimental changes in a single submission.** Three simultaneous changes (MegaDescriptor + RootSIFT + LNBNN) made diagnosis require an additional submission (V2.1).

2. **LNBNN with dense candidate pools is structurally broken.** When K=100 candidates each contribute ~1000 descriptors, the background estimate comes from 100k within-class descriptors — the background distance is so small that most deltas are ≤ 0 and scores collapse to zero.

3. **BFMatcher ratio test is better calibrated than LNBNN for this setting.** BFMatcher produces scores in a range that works with V1.3's thresholds. LNBNN does not.

4. **Score scale matters for clustering.** The clustering thresholds must be recalibrated whenever the ensemble score distribution changes. Carrying V1.3 thresholds into V2.0 was the proximate cause of the disaster.
