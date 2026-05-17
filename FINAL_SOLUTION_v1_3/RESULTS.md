# V1.3 Results — SAM3 Keypoint Mask Filtering

**Score: 0.32528**
**Previous best (V1): 0.30655**
**Improvement: +0.01873 absolute (+6.1% relative)**

---

## What V1.3 Does Differently from V1

SIFT runs on the **original image** (unchanged pixels). After keypoint detection, any keypoint whose `(x, y)` coordinate falls on a background pixel is discarded. Only animal-region keypoints are passed to matching.

```
Original image → SIFT detectAndCompute → all keypoints
                                              ↓
                              get_seg_mask(image_path)
                                              ↓
                         keep only: seg_mask[y, x] == 1
                                              ↓
                              animal keypoints only → matching
```

## What V1.2 Did Wrong (score 0.26330)

V1.2 painted the background white before feeding to SIFT. This created strong edges at the animal/white boundary → SIFT locked onto those boundary keypoints → boundary shape varies with pose/crop → false matches between different animals → score dropped.

---

## Cache Coverage

```
LynxID2025         train    0%   ⚠  (no change vs V1 for this species)
LynxID2025         test     0%   ⚠
SalamanderID2025   train  100%
SalamanderID2025   test   100%
SeaTurtleID2022    train  100%
SeaTurtleID2022    test   100%
TexasHornedLizards test   100%
```

LynxID2025 has 0% cache coverage → `get_seg_mask()` returns `None` → all keypoints kept → identical to V1 for that species. The gain comes entirely from the other 3 species.

---

## Version History

| Version | Score   | Approach |
|---------|---------|----------|
| V1      | 0.30655 | MiewID v3 + SIFT (baseline) |
| V1.2    | 0.26330 | + SAM3 white-background replacement — **hurt** |
| V1.3    | 0.32528 | + SAM3 keypoint mask filtering — **improved** |

---

## Files

- `build_notebook_v1_3.py` — build script (reads V1, patches, writes V1.3)
- `ensemble_global_local_reid_v1_3.ipynb` — submitted notebook
- `kernel-metadata.json` — Kaggle metadata

## Key Lesson

Painting background white before SIFT = bad (boundary artifact keypoints).
Filtering keypoints by mask after SIFT on original image = good (cleaner descriptors, no artifacts).
