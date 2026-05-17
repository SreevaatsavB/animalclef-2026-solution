"""
Threshold Tuning for AnimalCLEF 2026 Baseline
Grid search over KNOWN_THRESHOLD and UNKNOWN_CLUSTER_DIST
Reuses cached embeddings for fast experimentation.
"""

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('/Users/vaatsav/Desktop/New/AnimalCLEF_26')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from itertools import product

# Import similarity computation from src
from src.inference import compute_similarity_matrix

# Paths
BASE_DIR = Path('/Users/vaatsav/Desktop/New/AnimalCLEF_26')
EMBEDDINGS_DIR = BASE_DIR / 'baselines' / 'embeddings'
METADATA_PATH = BASE_DIR / 'data' / 'raw' / 'metadata.csv'
SUBMISSIONS_DIR = BASE_DIR / 'submissions'
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# Load metadata once
metadata = pd.read_csv(METADATA_PATH)

# Datasets with training data
DATASETS_WITH_TRAIN = ['LynxID2025', 'SalamanderID2025', 'SeaTurtleID2022']
ALL_DATASETS = DATASETS_WITH_TRAIN + ['TexasHornedLizards']

print("=" * 60)
print("AnimalCLEF 2026 — Threshold Tuning (Grid Search)")
print("=" * 60)
print(f"Metadata: {len(metadata)} rows")
print(f"Embeddings directory: {EMBEDDINGS_DIR}")
print()

# ============================================================
# Load all embeddings once
# ============================================================
print("Loading cached embeddings...")
embeddings_cache = {}

for dataset in ALL_DATASETS:
    test_path = EMBEDDINGS_DIR / f"{dataset}_test.npy"
    embeddings_cache[f"{dataset}_test"] = np.load(test_path)
    print(f"  ✓ {dataset}_test: {embeddings_cache[f'{dataset}_test'].shape}")

    if dataset in DATASETS_WITH_TRAIN:
        train_path = EMBEDDINGS_DIR / f"{dataset}_train.npy"
        embeddings_cache[f"{dataset}_train"] = np.load(train_path)
        print(f"  ✓ {dataset}_train: {embeddings_cache[f'{dataset}_train'].shape}")

print()

# ============================================================
# Grid Search Configuration
# ============================================================
KNOWN_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
UNKNOWN_CLUSTER_DISTS = [0.3, 0.4, 0.5, 0.6, 0.7]

print(f"Grid Search: {len(KNOWN_THRESHOLDS)} × {len(UNKNOWN_CLUSTER_DISTS)} = {len(KNOWN_THRESHOLDS) * len(UNKNOWN_CLUSTER_DISTS)} combinations")
print(f"  KNOWN_THRESHOLD: {KNOWN_THRESHOLDS}")
print(f"  UNKNOWN_CLUSTER_DIST: {UNKNOWN_CLUSTER_DISTS}")
print()

# ============================================================
# Helper Functions
# ============================================================

def run_phase1(dataset, known_threshold):
    """Phase 1: Match test images to known identities."""
    test_emb = embeddings_cache[f"{dataset}_test"]
    train_emb = embeddings_cache[f"{dataset}_train"]

    # Get train identities
    train_meta = metadata[(metadata['dataset'] == dataset) & (metadata['split'] == 'train')]
    train_ids = train_meta['identity'].values
    unique_ids = sorted(set(train_ids))

    # Compute prototypes (mean per identity, L2-normalized)
    prototypes = []
    for identity in unique_ids:
        idx = np.where(train_ids == identity)[0]
        proto = train_emb[idx].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto)
    prototypes = np.array(prototypes)

    # Cosine similarity
    sim_matrix = compute_similarity_matrix(test_emb, prototypes)

    # Matching
    max_sims = sim_matrix.max(axis=1)
    best_ids = sim_matrix.argmax(axis=1)

    matched_mask = max_sims >= known_threshold
    matched_indices = np.where(matched_mask)[0]
    unmatched_indices = np.where(~matched_mask)[0]

    # Cluster assignments for matched
    matched_clusters = best_ids[matched_indices]

    return matched_indices, unmatched_indices, matched_clusters, len(unique_ids)


def run_phase2(unmatched_embeddings, unknown_cluster_dist):
    """Phase 2: Cluster unknown identities."""
    n = len(unmatched_embeddings)

    if n == 0:
        return np.array([])
    elif n == 1:
        return np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=unknown_cluster_dist
        )
        cluster_labels = clustering.fit_predict(unmatched_embeddings)
        return cluster_labels


def generate_submission(known_threshold, unknown_cluster_dist):
    """Generate submission for given threshold combination."""
    all_results = {}
    cluster_offset = 0

    for dataset in ALL_DATASETS:
        test_meta = metadata[(metadata['dataset'] == dataset) & (metadata['split'] == 'test')]
        test_ids = test_meta['image_id'].values
        test_emb = embeddings_cache[f"{dataset}_test"]

        if dataset in DATASETS_WITH_TRAIN:
            # Phase 1: Known matching
            matched_idx, unmatched_idx, matched_clusters, n_known = run_phase1(dataset, known_threshold)

            # Phase 2: Unknown clustering
            if len(unmatched_idx) > 0:
                unmatched_emb = test_emb[unmatched_idx]
                unknown_clusters = run_phase2(unmatched_emb, unknown_cluster_dist)
                unknown_clusters = unknown_clusters + n_known  # Offset by known IDs
            else:
                unknown_clusters = np.array([])

            # Combine results
            for i, img_id in enumerate(test_ids):
                if i in matched_idx:
                    local_cluster = matched_clusters[np.where(matched_idx == i)[0][0]]
                    all_results[img_id] = f"cluster_{dataset}_{cluster_offset + local_cluster}"
                else:
                    local_idx = np.where(unmatched_idx == i)[0][0]
                    local_cluster = unknown_clusters[local_idx]
                    all_results[img_id] = f"cluster_{dataset}_{cluster_offset + local_cluster}"

            # Update offset for next dataset
            if len(unknown_clusters) > 0:
                cluster_offset += n_known + unknown_clusters.max() + 1
            else:
                cluster_offset += n_known

        else:
            # TexasHornedLizards: Pure zero-shot clustering (Phase 2 only)
            unknown_clusters = run_phase2(test_emb, unknown_cluster_dist)

            for i, img_id in enumerate(test_ids):
                local_cluster = unknown_clusters[i]
                all_results[img_id] = f"cluster_{dataset}_{cluster_offset + local_cluster}"

            cluster_offset += unknown_clusters.max() + 1

    # Create submission DataFrame
    submission_df = pd.DataFrame([
        {'image_id': img_id, 'cluster': cluster}
        for img_id, cluster in all_results.items()
    ])

    # Sort by image_id to match expected format
    submission_df = submission_df.sort_values('image_id').reset_index(drop=True)

    return submission_df


# ============================================================
# Grid Search
# ============================================================
print("Starting grid search...")
print()

results = []
best_score = None
best_config = None

for i, (kt, ucd) in enumerate(product(KNOWN_THRESHOLDS, UNKNOWN_CLUSTER_DISTS), 1):
    print(f"[{i}/{len(KNOWN_THRESHOLDS) * len(UNKNOWN_CLUSTER_DISTS)}] Testing KNOWN={kt:.1f}, UNKNOWN={ucd:.1f}", end=' ')

    # Generate submission
    submission_df = generate_submission(kt, ucd)

    # Stats
    n_clusters = submission_df['cluster'].nunique()

    # Save submission
    filename = f"baseline_kt{kt:.1f}_ucd{ucd:.1f}.csv"
    filepath = SUBMISSIONS_DIR / filename
    submission_df.to_csv(filepath, index=False)

    print(f"→ {n_clusters} clusters, saved to {filename}")

    results.append({
        'known_threshold': kt,
        'unknown_cluster_dist': ucd,
        'n_clusters': n_clusters,
        'filename': filename
    })

print()
print("=" * 60)
print("Grid Search Complete!")
print("=" * 60)

# Summary table
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string(index=False))

# Pivot table for visualization
pivot = results_df.pivot(index='known_threshold', columns='unknown_cluster_dist', values='n_clusters')
print("\nCluster Counts (KNOWN_THRESHOLD × UNKNOWN_CLUSTER_DIST):")
print(pivot.to_string())

print(f"\nAll {len(results)} submissions saved to {SUBMISSIONS_DIR}/")
print("\nNext steps:")
print("  1. Submit each CSV to the competition leaderboard")
print("  2. Compare scores to find optimal thresholds")
print("  3. Use best configuration for final submission")
