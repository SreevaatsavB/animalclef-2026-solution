# AnimalCLEF 2026 — Baseline Architecture: 2-Phase Clustering

## Overview

A two-stage clustering pipeline that assigns each test image to an individual animal cluster by first attempting to match against known identities, then grouping unmatched images into new identity clusters.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BASELINE PIPELINE                             │
└─────────────────────────────────────────────────────────────────────┘

                          INPUT DATA
                              ↓
        ┌─────────────────────────────────────────────────┐
        │  4 Datasets (2,409 test images total)          │
        │  • LynxID2025: 946 images                       │
        │  • SalamanderID2025: 689 images                 │
        │  • SeaTurtleID2022: 500 images                  │
        │  • TexasHornedLizards: 274 images (zero-shot)   │
        └─────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────┐
        │     STAGE 0: FEATURE EXTRACTION                 │
        │  (MegaDescriptor-L-384 on MPS)                  │
        │  • Load model: hf-hub:BVRA/MegaDescriptor-L-384 │
        │  • Input: Resize(384) → Normalize → Embed       │
        │  • Output: 1536-dim vectors (L2-normalized)     │
        │  • Cache: baselines/embeddings/*.npy            │
        └─────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────┐
        │  STAGE 1: KNOWN IDENTITY MATCHING               │
        │  (For datasets with training data)              │
        ├─────────────────────────────────────────────────┤
        │  Per dataset:                                   │
        │  1. Group train embeddings by identity          │
        │  2. Compute L2-normalized mean prototypes       │
        │  3. Cosine similarity: test_emb @ proto.T       │
        │  4. If max_sim ≥ threshold (0.5):              │
        │     → Assign to best-matching identity          │
        │  5. Track matched vs unmatched images           │
        └─────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────┐
        │  STAGE 2: UNKNOWN IDENTITY CLUSTERING           │
        │  (For unmatched + zero-shot images)             │
        ├─────────────────────────────────────────────────┤
        │  Per dataset (unmatched images):                │
        │  • If 0-1 images: each is own cluster           │
        │  • Else: AgglomerativeClustering                │
        │    - metric='cosine'                            │
        │    - linkage='average'                          │
        │    - distance_threshold=0.5                     │
        │  → Generate new cluster IDs                     │
        └─────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────┐
        │  STAGE 3: SUBMISSION GENERATION                 │
        ├─────────────────────────────────────────────────┤
        │  Build cluster mappings:                        │
        │  • Known: cluster_{dataset}_{idx}               │
        │  • Unknown: cluster_{dataset}_{N_known + id}    │
        │  Write CSV: image_id, cluster                   │
        └─────────────────────────────────────────────────┘
                              ↓
                        OUTPUT: CSV
        submissions/baseline_2phase_clustering.csv
```

---

## Phase 1: Known Identity Matching

```
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 1: MATCHING                          │
└─────────────────────────────────────────────────────────────┘

TEST EMBEDDINGS                          PROTOTYPES
[test_1]                                [Identity_0]
[test_2]           Cosine Similarity    [Identity_1]
[test_3] ────────→    @              ──→[Identity_2]
[test_4]           (n_test × n_ids)      [...]
[...]                                 [Identity_N]

              ↓ (compute argmax per row)

        MATCHING RESULTS
        ┌──────────────────────────┐
        │ test_1 → Identity_5 ✓    │  max_sim=0.67 ≥ 0.5
        │ test_2 → UNMATCHED ✗     │  max_sim=0.32 < 0.5
        │ test_3 → Identity_12 ✓   │  max_sim=0.71 ≥ 0.5
        │ test_4 → UNMATCHED ✗     │  max_sim=0.41 < 0.5
        │ ...                      │
        └──────────────────────────┘
```

### Results by Dataset

| Dataset | Threshold | Matched | Unmatched | Match Rate |
|---------|-----------|---------|-----------|------------|
| **LynxID2025** | 0.5 | 740/946 | 206 | **78%** |
| **SalamanderID2025** | 0.5 | 677/689 | 12 | **98%** |
| **SeaTurtleID2022** | 0.5 | 189/500 | 311 | **38%** |
| **TexasHornedLizards** | N/A | 0/274 | 274 | **0%** (no training data) |

---

## Phase 2: Unknown Identity Clustering

```
┌─────────────────────────────────────────────────────────────┐
│         PHASE 2: UNKNOWN CLUSTERING                          │
└─────────────────────────────────────────────────────────────┘

INPUT: Unmatched test embeddings + TexasHornedLizards test set
       (each represented as 1536-dim vector)

                         COSINE DISTANCE MATRIX
                         test_i  test_j  test_k ...
                test_i [  0      0.15    0.42   ...]
                test_j [ 0.15     0      0.38   ...]
                test_k [ 0.42    0.38     0      ...]
                ...    [...     ...      ...    ...]

                              ↓

              AGGLOMERATIVE CLUSTERING
              • Metric: Cosine distance
              • Linkage: Average
              • Distance threshold: 0.5

                              ↓

                    CLUSTER DENDROGRAM
                              ┌─────────────┐
                              │             │
                         ┌────┴──┐      ┌──┴────┐
                         │       │      │       │
                      ┌──┴┐  ┌──┴┐  ┌──┴┐  ┌──┴┐
                     (A) (B) (C) (D) (E) (F) (G) (H)
                      │   │   │   │   │   │   │   │
                    test1,test3 test5 test2... test8

                              ↓

                      NEW CLUSTER LABELS
                Cluster_0: {test_1, test_3}
                Cluster_1: {test_5}
                Cluster_2: {test_2, test_7, test_9}
                ...
```

### Results by Dataset

| Dataset | Unmatched | New Clusters | Avg Cluster Size |
|---------|-----------|--------------|------------------|
| **LynxID2025** | 206 | 158 | 1.30 |
| **SalamanderID2025** | 12 | 11 | 1.09 |
| **SeaTurtleID2022** | 311 | 228 | 1.36 |
| **TexasHornedLizards** | 274 | 79 | 3.47 |

---

## Cluster Naming Convention

```
┌─────────────────────────────────────────────────────────────┐
│              CLUSTER ID GENERATION                           │
└─────────────────────────────────────────────────────────────┘

FOR MATCHED IMAGES (Phase 1):
  cluster_<dataset>_<identity_index>

  Example: cluster_LynxID2025_0  (first known identity in Lynx)
           cluster_SalamanderID2025_45  (46th known identity)

FOR UNMATCHED IMAGES (Phase 2):
  cluster_<dataset>_<n_known + cluster_id>

  Example: cluster_LynxID2025_58  (58 known IDs, cluster 0 → idx 58)
           cluster_SalamanderID2025_295  (294 known IDs, cluster 1 → idx 295)
           cluster_TexasHornedLizards_0  (0 known IDs, cluster 0 → idx 0)
```

---

## Data Flow Diagram

```
┌──────────────────┐
│   RAW IMAGES     │
│  (15,483 total)  │
└────────┬─────────┘
         │
         ↓
┌──────────────────────────────────────┐
│  TRAIN IMAGES (for 3 datasets)       │
│  • LynxID2025: 2,957 images (77 IDs) │
│  • SalamanderID2025: 1,388 (587 IDs) │
│  • SeaTurtleID2022: 8,729 (438 IDs)  │
│  • TexasHornedLizards: 0 (ZERO-SHOT) │
└────────┬─────────────────────────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ↓                                 ↓
┌────────────────────┐         ┌──────────────────────┐
│ PROTOTYPE CREATION │         │ TEST EMBEDDINGS      │
│ (L2-normalized     │         │ (1536-dim vectors)   │
│  mean per ID)      │         │ 2,409 images         │
└────────┬───────────┘         └──────────┬───────────┘
         │                                │
         └──────────────────┬─────────────┘
                            │
                            ↓
                ┌──────────────────────────┐
                │ COSINE SIMILARITY MATRIX │
                │ (test_emb @ proto.T)     │
                └────────────┬─────────────┘
                             │
                ┌────────────┴─────────────┐
                │                          │
                ↓                          ↓
        ┌──────────────┐        ┌────────────────┐
        │   MATCHED    │        │   UNMATCHED    │
        │  1,606 imgs  │        │   803 images   │
        │  (425 IDs)   │        └────────┬───────┘
        └──────┬───────┘                 │
               │                         │
               │        ┌────────────────┤
               │        │                │
               ↓        ↓                ↓
        ┌─────────────────────────────────────────┐
        │  AGGLOMERATIVE CLUSTERING               │
        │  (cosine distance, threshold=0.5)       │
        │  → 476 new clusters                     │
        └────────────┬────────────────────────────┘
                     │
                     ↓
        ┌──────────────────────────────────┐
        │  FINAL CLUSTER ASSIGNMENT         │
        │  (901 total unique clusters)      │
        │  ✅ 2,409 test images assigned    │
        └────────────┬─────────────────────┘
                     │
                     ↓
        ┌──────────────────────────────────┐
        │  OUTPUT CSV                       │
        │  image_id, cluster                │
        │  (2,410 lines: 1 header + 2,409)  │
        └──────────────────────────────────┘
```

---

## Component Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    KEY COMPONENTS                            │
└─────────────────────────────────────────────────────────────┘

1. MODEL
   └─ MegaDescriptor-L-384 (BVRA/MegaDescriptor-L-384)
      • 384→1536 dimensional embeddings
      • Pretrained on large-scale wildlife images
      • Device: MPS (Apple Silicon GPU)

2. SIMILARITY COMPUTATION
   └─ compute_similarity_matrix() [src/inference.py:49]
      • Cosine similarity: test_emb @ prototypes.T
      • Output: [n_test, n_prototypes] matrix

3. CLUSTERING ALGORITHM
   └─ sklearn.cluster.AgglomerativeClustering
      • metric='cosine'
      • linkage='average'
      • distance_threshold=0.5

4. EMBEDDING CACHE
   └─ baselines/embeddings/
      • {dataset}_train.npy: Training embeddings
      • {dataset}_test.npy: Test embeddings
      • Enables fast re-runs without re-extraction

5. METADATA
   └─ data/raw/metadata.csv
      • Image IDs, dataset labels, splits
      • 15,483 total rows
```

---

## Hyperparameters

```
┌─────────────────────────────────────────────────────────────┐
│                  TUNABLE PARAMETERS                          │
└─────────────────────────────────────────────────────────────┘

FEATURE EXTRACTION
  ├─ MODEL_NAME: 'hf-hub:BVRA/MegaDescriptor-L-384'
  ├─ INPUT_SIZE: 384 (RGB image size)
  └─ BATCH_SIZE: 32 (embedding extraction batches)

PHASE 1: KNOWN MATCHING
  └─ KNOWN_THRESHOLD: 0.5
     (cosine similarity threshold for known identity match)

PHASE 2: UNKNOWN CLUSTERING
  └─ UNKNOWN_CLUSTER_DIST: 0.5
     (cosine distance threshold for agglomerative clustering)
```

---

## Performance Metrics

```
┌─────────────────────────────────────────────────────────────┐
│             BASELINE PERFORMANCE SUMMARY                     │
└─────────────────────────────────────────────────────────────┘

MATCHING PERFORMANCE
  • Total known matches: 1,606 / 2,409 (67%)
  • Total unmatched: 803 / 2,409 (33%)
  • Best dataset: SalamanderID2025 (98% match rate)
  • Worst dataset: TexasHornedLizards (0% - no training data)

CLUSTERING PERFORMANCE
  • Total unmatched images: 803
  • Unknown clusters generated: 476
  • Average cluster size (unknown): 1.69
  • Total clusters (known + unknown): 901

COMPUTATIONAL EFFICIENCY
  • Feature extraction time: ~13 min (SeaTurtleID2022_train)
  • Total embeddings cached: 7 files (~190 MB)
  • Phase 1 + 2 execution: <5 minutes
  • Full pipeline (with caching): ~14 minutes total

SUBMISSION STATS
  • Output file: baseline_2phase_clustering.csv
  • File size: 73 KB
  • Rows: 2,409 images + 1 header
  • ✅ Validation: All sanity checks passed
```

---

## Execution Flow

```
python baselines/run_baseline.py

  1. Load metadata (15,483 rows)
  2. Detect device (MPS)
  3. Load MegaDescriptor-L-384 model

  FOR EACH DATASET:
    a. Load/extract test embeddings
    b. Load/extract train embeddings (if available)
    c. PHASE 1: Match test to known identities
    d. PHASE 2: Cluster unmatched with agglomerative clustering
    e. Build cluster mappings

  4. Merge all results → 901 unique clusters
  5. Generate submission CSV (2,409 rows)
  6. Validate output (5 sanity checks)
  7. Print per-dataset summary statistics
```

---

## Extensions & Future Work

```
POTENTIAL IMPROVEMENTS

1. THRESHOLD TUNING
   • Currently: KNOWN_THRESHOLD = 0.5 (fixed)
   • Option: Dataset-specific thresholds (0.3-0.7)
   • Option: Validation-based threshold selection

2. METRIC LEARNING
   • Currently: Pre-trained MegaDescriptor
   • Option: Fine-tune on wildlife re-identification data
   • Option: Metric learning (contrastive/triplet loss)

3. ENSEMBLE CLUSTERING
   • Currently: Single agglomerative clustering
   • Option: Multiple linkage methods (ward, complete, single)
   • Option: Distance threshold sweep + voting

4. ADVANCED ALGORITHMS
   • Currently: Agglomerative clustering
   • Option: DBSCAN with learned epsilon per dataset
   • Option: Neural clustering (deep embedded clustering)
   • Option: Graph-based clustering on similarity graphs

5. CONFIDENCE SCORING
   • Currently: Hard assignments only
   • Option: Soft cluster probabilities
   • Option: Confidence calibration for borderline cases
```

---

## File Structure

```
AnimalCLEF_26/
├── baselines/
│   ├── ARCHITECTURE.md              ← You are here
│   ├── run_baseline.py              ← Main pipeline script
│   └── embeddings/                  ← Feature cache
│       ├── LynxID2025_train.npy
│       ├── LynxID2025_test.npy
│       ├── SalamanderID2025_train.npy
│       ├── SalamanderID2025_test.npy
│       ├── SeaTurtleID2022_train.npy
│       ├── SeaTurtleID2022_test.npy
│       └── TexasHornedLizards_test.npy
│
├── submissions/
│   └── baseline_2phase_clustering.csv  ← OUTPUT
│
├── data/
│   └── raw/
│       ├── metadata.csv
│       └── sample_submission.csv
│
└── src/
    ├── inference.py          (compute_similarity_matrix)
    └── models.py             (model loading utilities)
```

---

**Last Updated:** Feb 13, 2025
**Status:** ✅ Baseline Implementation Complete
