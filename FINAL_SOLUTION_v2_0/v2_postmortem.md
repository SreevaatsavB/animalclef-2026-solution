# V2.0 Post-Mortem (score: 0.10691 vs V1.3: 0.32528)

## What V2.0 added (all at once)

1. **MegaDescriptor-L-384** — Swin-L fused with MiewID → 3688-dim global embeddings
2. **RootSIFT** — L1-norm + sqrt on SIFT descriptors before keypoint mask filter
3. **LNBNN scoring** — HotSpotter-style matching replacing BFMatcher ratio test

## Why simultaneous changes are dangerous

Can't isolate which change caused the failure from a single score.
Score 0.10691 is so low it implies structural failure, not marginal regression.

## Root cause hypothesis: MegaDescriptor corrupting global similarity

**Why global_sim is the single point of failure:**
- Used for `global_weight * global_sim` in ensemble → 65-70% of total score
- Used for top-K candidate selection for LNBNN
- If global_sim is garbage, BOTH paths fail simultaneously

**Why MegaDescriptor is the prime suspect:**
- 1536/3688 = 42% of the fused embedding comes from MegaDescriptor
- If MegaDescriptor features are off-distribution for these species/conditions,
  cosine similarities in fused space can be WORSE than MiewID alone
- MiewID was finetuned on conservation datasets; MegaDescriptor on 37k+ wildlife individuals,
  but the image conditions (lighting, angle, quality) may differ substantially
- L2-renormalization after concat means MegaDescriptor's 42% share can invert rankings

**Why RootSIFT/LNBNN are unlikely to cause 0.10 alone:**
- Local weight is only 30-35% of ensemble
- Even with scale completely wrong, local scores would just be uniform → degrade to global-only
- Global-only (MiewID) should score ~0.30+ based on V1 baseline

## Other possibilities (lower probability)

- MegaDescriptor returns NaN/inf → fused embeddings are NaN → all cosine sims undefined
- MEGA_BATCH_SIZE=32 still OOM'd → features extracted incorrectly
- LNBNN scale=20 completely wrong for this descriptor space → local scores all near 0 or 1
  (but this still shouldn't drop to 0.10 since global weight is 65-70%)

## What "blacked out segmentation" in submission description means

Likely just the notebook output displaying SAM3-segmented image thumbnails (which have
white/black backgrounds from the segmentation). NOT a processing bug — our code never
applies segmentation to images before feature extraction.

## Diagnostic plan

| Submission | Changes from V1.3 | Purpose |
|---|---|---|
| V2.1 | RootSIFT + LNBNN only (no MegaDescriptor) | Isolate MegaDescriptor vs RootSIFT/LNBNN |
| V2.2 | MegaDescriptor only (no RootSIFT, no LNBNN) | Confirm MegaDescriptor impact in isolation |
| V2.3 | RootSIFT only (no MegaDescriptor, no LNBNN) | Confirm RootSIFT impact in isolation |

V2.1 is the most informative single submission. Build it first.

## V2.0 bugs fixed before submission (did not cause the score drop)

These were implementation bugs caught during code review, NOT runtime failures:
1. Missing `num_classes=0` in `timm.create_model` — would return class logits not embeddings
2. Self-image included in LNBNN db_descs_list — distorts background distance
3. `np.add.at` accumulation — slow but correct; replaced with GPU `scatter_add_`
4. `np.argsort` → `np.argpartition` — optimization only
5. Python loop → vectorised `np.maximum` — optimization only

Despite bug fixes, score still dropped catastrophically. The issue is likely the
MegaDescriptor approach itself, not implementation bugs.
