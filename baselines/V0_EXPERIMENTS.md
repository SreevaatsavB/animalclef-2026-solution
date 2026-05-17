# Baseline V0 — Threshold Tuning Experiments

## Version Information

**Version:** v0
**Date:** February 13, 2025
**Method:** 2-Phase Clustering (Known Matching + Unknown Clustering)
**Model:** MegaDescriptor-L-384
**Embeddings:** Cached in `baselines/embeddings/` (reused across all experiments)

---

## Baseline Configuration

```python
MODEL_NAME = 'hf-hub:BVRA/MegaDescriptor-L-384'
INPUT_SIZE = 384
BATCH_SIZE = 32

# Phase 1: Known Identity Matching
KNOWN_THRESHOLD = 0.5        # Cosine similarity threshold

# Phase 2: Unknown Identity Clustering
UNKNOWN_CLUSTER_DIST = 0.5   # Agglomerative clustering distance threshold
```

---

## Leaderboard Results

### 🏆 Submission 1: Baseline (BEST)
**File:** `v0_baseline_kt0.5_ucd0.5.csv`
**Config:** KNOWN_THRESHOLD=0.5, UNKNOWN_CLUSTER_DIST=0.5
**Clusters:** 901
**Score:** **0.13663**
**Rank:** **#1** 🥇
**Status:** ✅ Best performing configuration

**Per-Dataset Breakdown:**
| Dataset | Known Matched | Unknown Clustered | Total Clusters |
|---------|---------------|-------------------|----------------|
| LynxID2025 | 740/946 (78%) | 206 → 158 clusters | 216 |
| SalamanderID2025 | 677/689 (98%) | 12 → 11 clusters | 305 |
| SeaTurtleID2022 | 189/500 (38%) | 311 → 228 clusters | 301 |
| TexasHornedLizards | 0/274 (0%) | 274 → 79 clusters | 79 |
| **TOTAL** | **1,606** | **803 → 476 clusters** | **901** |

---

### ❌ Submission 2: Conservative Thresholds
**File:** `v0_kt0.7_ucd0.4.csv`
**Config:** KNOWN_THRESHOLD=0.7, UNKNOWN_CLUSTER_DIST=0.4
**Clusters:** 1,404
**Score:** **0.12406**
**Rank:** Lower than baseline
**Status:** ❌ Performance decreased (more clusters = worse)

**Analysis:**
- Higher KNOWN_THRESHOLD (0.7) → Fewer test images matched to known identities
- Lower UNKNOWN_CLUSTER_DIST (0.4) → Less aggressive clustering, more individual clusters
- Result: 56% more clusters than baseline (1,404 vs 901)
- **Conclusion:** Too conservative — splits known individuals, hurts BAKS

---

### ❌ Submission 3: Aggressive Thresholds
**File:** `v0_kt0.3_ucd0.7.csv`
**Config:** KNOWN_THRESHOLD=0.3, UNKNOWN_CLUSTER_DIST=0.7
**Clusters:** 613
**Score:** **0.10756**
**Rank:** Lowest among tested
**Status:** ❌ Performance decreased significantly (fewer clusters = worse)

**Analysis:**
- Lower KNOWN_THRESHOLD (0.3) → More test images matched to known identities (over-matching)
- Higher UNKNOWN_CLUSTER_DIST (0.7) → Very aggressive clustering, merges distinct individuals
- Result: 32% fewer clusters than baseline (613 vs 901)
- **Conclusion:** Too aggressive — merges distinct individuals, hurts BAUS

---

## Summary of Findings

```
┌─────────────────────────────────────────────────────────────┐
│           V0 THRESHOLD TUNING RESULTS                        │
└─────────────────────────────────────────────────────────────┘

Configuration          Clusters    Score      Δ from Baseline
────────────────────────────────────────────────────────────────
kt=0.5, ucd=0.5 ✅       901      0.13663    BASELINE (BEST)
kt=0.7, ucd=0.4          1,404    0.12406    -9.2% ❌
kt=0.3, ucd=0.7          613      0.10756    -21.3% ❌

KEY INSIGHTS:
  • Baseline (kt=0.5, ucd=0.5) appears optimal
  • More conservative → more clusters → lower score
  • More aggressive → fewer clusters → lower score
  • Sweet spot: ~900 clusters balances BAKS and BAUS
```

---

## Evaluation Metric: Geometric Mean

The competition uses **geometric mean of BAKS × BAUS**:

- **BAKS:** Balanced Accuracy on Known Subjects (identities in training set)
- **BAUS:** Balanced Accuracy on Unknown Subjects (novel identities)

**Why the baseline works:**
1. **kt=0.5** provides balanced known/unknown split
   - Not too strict: matches genuine known individuals (good BAKS)
   - Not too lenient: avoids false matches of unknowns to known (good BAUS)

2. **ucd=0.5** balances clustering granularity
   - Not too conservative: merges same individuals (good BAUS)
   - Not too aggressive: keeps distinct individuals separate (good BAUS)

3. **Geometric mean** penalizes imbalance
   - Too conservative: Hurts BAKS (splits known individuals)
   - Too aggressive: Hurts BAUS (merges distinct unknowns)
   - Baseline achieves best balance

---

## Hyperparameter Sensitivity Analysis

```
KNOWN_THRESHOLD (Phase 1 Matching):
  0.3 → Over-matching (assigns unknowns to known IDs)
  0.5 → Balanced ✅
  0.7 → Under-matching (treats known IDs as unknown)

UNKNOWN_CLUSTER_DIST (Phase 2 Clustering):
  0.3 → Under-clustering (splits individuals)
  0.5 → Balanced ✅
  0.7 → Over-clustering (merges distinct animals)

INTERACTION EFFECT:
  Both thresholds work together:
  • High kt + Low ucd → Many small clusters
  • Low kt + High ucd → Few large clusters
  • Balanced kt & ucd → Optimal granularity
```

---

## Next Steps (Future Versions)

### V1 Potential Improvements

1. **Per-Dataset Thresholds**
   - Current: Single global threshold for all datasets
   - Proposed: Dataset-specific thresholds based on characteristics
     - LynxID2025: Good match rate (78%) → keep kt=0.5
     - SalamanderID2025: Excellent match rate (98%) → try kt=0.4
     - SeaTurtleID2022: Poor match rate (38%) → try kt=0.6
     - TexasHornedLizards: Zero-shot only → optimize ucd separately

2. **Advanced Clustering Algorithms**
   - DBSCAN with learned epsilon
   - Graph-based clustering on similarity graphs
   - Neural clustering (Deep Embedded Clustering)
   - Ensemble of multiple clustering methods

3. **Metric Learning Fine-tuning**
   - Fine-tune MegaDescriptor on wildlife re-identification
   - Learn dataset-specific embeddings
   - Contrastive or triplet loss training

4. **Feature Ensemble**
   - Combine multiple backbones (MegaDescriptor + DINOv2 + others)
   - Multi-scale features (global + local patches)
   - Attention-weighted feature fusion

5. **Confidence-Based Assignment**
   - Soft clustering with probabilities
   - Threshold based on confidence calibration
   - Handle borderline cases explicitly

---

## Files in This Version

```
baselines/
├── V0_EXPERIMENTS.md              ← This file
├── ARCHITECTURE.md                ← Overall architecture documentation
├── run_baseline.py                ← Original baseline script
├── tune_thresholds.py             ← Grid search script (used to generate v0 experiments)
└── embeddings/                    ← Cached embeddings (reused across all v0)
    ├── LynxID2025_train.npy
    ├── LynxID2025_test.npy
    ├── SalamanderID2025_train.npy
    ├── SalamanderID2025_test.npy
    ├── SeaTurtleID2022_train.npy
    ├── SeaTurtleID2022_test.npy
    └── TexasHornedLizards_test.npy

submissions/
├── v0_baseline_kt0.5_ucd0.5.csv   ← Score: 0.13663 (Rank #1) ✅
├── v0_kt0.7_ucd0.4.csv            ← Score: 0.12406 ❌
└── v0_kt0.3_ucd0.7.csv            ← Score: 0.10756 ❌
```

---

## Reproducibility

To reproduce the v0 baseline:

```bash
# Activate environment
source venv_animalclef2026/bin/activate

# Run baseline (generates embeddings and submission)
python baselines/run_baseline.py

# Output: submissions/baseline_2phase_clustering.csv
# (renamed to v0_baseline_kt0.5_ucd0.5.csv)
```

To reproduce the threshold tuning:

```bash
# Run grid search (reuses cached embeddings)
python baselines/tune_thresholds.py

# Generates 25 submissions in ~30 seconds
# Manually tested: kt0.7_ucd0.4 and kt0.3_ucd0.7
```

---

**Last Updated:** Feb 13, 2025
**Status:** ✅ V0 Complete — Baseline is optimal among tested configurations
**Next:** Explore V1 improvements (per-dataset tuning, advanced algorithms, metric learning)
