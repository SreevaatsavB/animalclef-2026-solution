"""
Baseline Solution: 2-Phase Clustering for AnimalCLEF 2026

Phase 1: Match test images to known identities via cosine similarity to prototypes.
Phase 2: Cluster remaining (unknown) test images via agglomerative clustering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.inference import compute_similarity_matrix

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-L-384"
INPUT_SIZE = 384
KNOWN_THRESHOLD = 0.5        # cosine sim threshold for "known" match
UNKNOWN_CLUSTER_DIST = 0.5   # agglomerative clustering distance threshold
BATCH_SIZE = 32

DATASETS_WITH_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
ZERO_SHOT_DATASETS = ["TexasHornedLizards"]
ALL_DATASETS = DATASETS_WITH_TRAIN + ZERO_SHOT_DATASETS

EMBEDDINGS_DIR = ROOT / "baselines" / "embeddings"
SUBMISSION_PATH = ROOT / "submissions" / "baseline_2phase_clustering.csv"
METADATA_PATH = ROOT / "data" / "raw" / "metadata.csv"


# ── Step 1: Feature Extraction ────────────────────────────────────────────────

class ImageDataset(Dataset):
    """Simple dataset that loads images and applies transforms."""

    def __init__(self, image_paths: list[str], root: Path, input_size: int = 384):
        self.image_paths = image_paths
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.root / self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, 0  # label unused, but DeepFeatures expects (image, label)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_and_cache_embeddings(
    metadata: pd.DataFrame,
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Extract embeddings for every dataset split, caching to disk.
    Returns: {dataset: {"train": ndarray|None, "test": ndarray, "train_paths": [...], "test_paths": [...]}}
    """
    # Check if all cache files already exist
    all_cached = True
    for ds in ALL_DATASETS:
        test_cache = EMBEDDINGS_DIR / f"{ds}_test.npy"
        if not test_cache.exists():
            all_cached = False
            break
        if ds in DATASETS_WITH_TRAIN:
            train_cache = EMBEDDINGS_DIR / f"{ds}_train.npy"
            if not train_cache.exists():
                all_cached = False
                break

    # Load model only if we need to extract
    model = None
    if not all_cached:
        print(f"Loading model {MODEL_NAME} on {device}...")
        from wildlife_tools.features.deep import DeepFeatures
        backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        model = DeepFeatures(model=backbone, device=device, batch_size=BATCH_SIZE, num_workers=0)
        print("Model loaded.")

    results = {}
    for ds in ALL_DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        ds_meta = metadata[metadata["dataset"] == ds]

        # -- Test embeddings --
        test_meta = ds_meta[ds_meta["split"] == "test"].reset_index(drop=True)
        test_paths = test_meta["path"].tolist()
        test_cache = EMBEDDINGS_DIR / f"{ds}_test.npy"

        if test_cache.exists():
            print(f"  Loading cached test embeddings: {test_cache}")
            test_emb = np.load(test_cache)
        else:
            print(f"  Extracting test embeddings ({len(test_paths)} images)...")
            test_dataset = ImageDataset(test_paths, ROOT / "data" / "raw", INPUT_SIZE)
            test_emb = model(test_dataset)
            np.save(test_cache, test_emb)
            print(f"  Cached to {test_cache}")

        # -- Train embeddings (if applicable) --
        train_emb = None
        train_paths = []
        train_identities = []
        if ds in DATASETS_WITH_TRAIN:
            train_meta = ds_meta[ds_meta["split"] == "train"].reset_index(drop=True)
            train_paths = train_meta["path"].tolist()
            train_identities = train_meta["identity"].tolist()
            train_cache = EMBEDDINGS_DIR / f"{ds}_train.npy"

            if train_cache.exists():
                print(f"  Loading cached train embeddings: {train_cache}")
                train_emb = np.load(train_cache)
            else:
                print(f"  Extracting train embeddings ({len(train_paths)} images)...")
                train_dataset = ImageDataset(train_paths, ROOT / "data" / "raw", INPUT_SIZE)
                train_emb = model(train_dataset)
                np.save(train_cache, train_emb)
                print(f"  Cached to {train_cache}")

        results[ds] = {
            "train_emb": train_emb,
            "test_emb": test_emb,
            "test_image_ids": test_meta["image_id"].tolist(),
            "train_identities": train_identities,
        }
        print(f"  Test: {test_emb.shape}, Train: {train_emb.shape if train_emb is not None else 'N/A'}")

    return results


# ── Step 2: Phase 1 — Known Identity Matching ────────────────────────────────

def phase1_known_matching(
    train_emb: np.ndarray,
    train_identities: list[str],
    test_emb: np.ndarray,
    threshold: float,
) -> tuple[dict[int, str], list[int]]:
    """
    Match test images to known identities via prototype similarity.
    Returns:
        matched: {test_idx: identity_name}
        unmatched: [test_idx, ...]
    """
    # Build prototypes: mean embedding per identity, L2-normalized
    unique_ids = sorted(set(train_identities))
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    train_id_indices = np.array([id_to_idx[tid] for tid in train_identities])

    prototypes = np.zeros((len(unique_ids), train_emb.shape[1]), dtype=np.float32)
    for i, uid in enumerate(unique_ids):
        mask = train_id_indices == i
        proto = train_emb[mask].mean(axis=0)
        prototypes[i] = proto
    # L2 normalize prototypes
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
    prototypes = prototypes / norms

    # Cosine similarity: test_emb @ prototypes.T
    sim_matrix = compute_similarity_matrix(test_emb, prototypes, metric="cosine")

    matched = {}
    unmatched = []
    for i in range(len(test_emb)):
        max_sim = sim_matrix[i].max()
        if max_sim >= threshold:
            best_idx = sim_matrix[i].argmax()
            matched[i] = unique_ids[best_idx]
        else:
            unmatched.append(i)

    return matched, unmatched


# ── Step 3: Phase 2 — Unknown Identity Clustering ────────────────────────────

def phase2_unknown_clustering(
    test_emb: np.ndarray,
    unmatched_indices: list[int],
    distance_threshold: float,
) -> dict[int, int]:
    """
    Cluster unmatched test images into new identity groups.
    Returns: {test_idx: cluster_id}
    """
    if len(unmatched_indices) == 0:
        return {}

    if len(unmatched_indices) == 1:
        return {unmatched_indices[0]: 0}

    sub_emb = test_emb[unmatched_indices]

    # L2 normalize for cosine distance
    norms = np.linalg.norm(sub_emb, axis=1, keepdims=True) + 1e-8
    sub_emb_norm = sub_emb / norms

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(sub_emb_norm)

    return {idx: int(lab) for idx, lab in zip(unmatched_indices, labels)}


# ── Step 4 & 5: Generate Submission & Sanity Checks ──────────────────────────

def build_submission(data: dict, metadata: pd.DataFrame) -> pd.DataFrame:
    """Build the full submission DataFrame."""
    rows = []

    for ds in ALL_DATASETS:
        info = data[ds]
        test_image_ids = info["test_image_ids"]
        matched = info.get("matched", {})
        unknown_clusters = info.get("unknown_clusters", {})

        # Build identity → cluster label mapping for known identities
        known_identities = sorted(set(matched.values())) if matched else []
        known_id_to_cluster = {
            uid: f"cluster_{ds}_{i}" for i, uid in enumerate(known_identities)
        }
        n_known = len(known_identities)

        n_known_matched = 0
        n_unknown_clustered = 0

        for test_idx, image_id in enumerate(test_image_ids):
            if test_idx in matched:
                cluster_label = known_id_to_cluster[matched[test_idx]]
                n_known_matched += 1
            elif test_idx in unknown_clusters:
                cluster_id = unknown_clusters[test_idx]
                cluster_label = f"cluster_{ds}_{n_known + cluster_id}"
                n_unknown_clustered += 1
            else:
                # Should not happen, but safety fallback: own cluster
                cluster_label = f"cluster_{ds}_solo_{test_idx}"

            rows.append({"image_id": image_id, "cluster": cluster_label})

        n_unknown_ids = len(set(unknown_clusters.values())) if unknown_clusters else 0
        total_clusters = n_known + n_unknown_ids
        print(f"\n{ds}:")
        print(f"  Known matched:   {n_known_matched} images → {n_known} identities")
        print(f"  Unknown grouped:  {n_unknown_clustered} images → {n_unknown_ids} clusters")
        print(f"  Total clusters:   {total_clusters}")

    submission = pd.DataFrame(rows)
    return submission


def sanity_check(submission: pd.DataFrame, metadata: pd.DataFrame):
    """Validate the submission."""
    test_meta = metadata[metadata["split"] == "test"]
    expected_ids = set(test_meta["image_id"].tolist())
    actual_ids = set(submission["image_id"].tolist())

    print(f"\n{'='*60}")
    print("Sanity Checks:")
    assert len(submission) == len(expected_ids), (
        f"Row count mismatch: {len(submission)} vs expected {len(expected_ids)}"
    )
    print(f"  [OK] Submission has {len(submission)} rows")

    assert actual_ids == expected_ids, (
        f"Missing IDs: {expected_ids - actual_ids}, Extra IDs: {actual_ids - expected_ids}"
    )
    print("  [OK] All test image_ids present, no extras")

    assert submission["image_id"].nunique() == len(submission), "Duplicate image_ids found!"
    print("  [OK] No duplicate image_ids")

    n_clusters = submission["cluster"].nunique()
    print(f"  Total unique clusters: {n_clusters}")

    # Load sample submission for comparison
    sample_sub = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv")
    sample_clusters = sample_sub["cluster"].nunique()
    print(f"  Sample submission clusters: {sample_clusters}")

    assert list(submission.columns) == ["image_id", "cluster"], (
        f"Wrong columns: {list(submission.columns)}"
    )
    print("  [OK] Correct column names (image_id, cluster)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("AnimalCLEF 2026 — Baseline: 2-Phase Clustering")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Metadata: {len(metadata)} rows")

    # Step 1: Extract embeddings
    data = extract_and_cache_embeddings(metadata, device)

    # Step 2 & 3: Phase 1 (known matching) + Phase 2 (unknown clustering)
    for ds in ALL_DATASETS:
        info = data[ds]
        test_emb = info["test_emb"]

        if ds in DATASETS_WITH_TRAIN:
            # Phase 1: known matching
            matched, unmatched = phase1_known_matching(
                info["train_emb"],
                info["train_identities"],
                test_emb,
                KNOWN_THRESHOLD,
            )
            info["matched"] = matched
            print(f"\n{ds} Phase 1: {len(matched)} matched, {len(unmatched)} unmatched")

            # Phase 2: cluster the unmatched
            unknown_clusters = phase2_unknown_clustering(
                test_emb, unmatched, UNKNOWN_CLUSTER_DIST
            )
            info["unknown_clusters"] = unknown_clusters
            n_new = len(set(unknown_clusters.values())) if unknown_clusters else 0
            print(f"{ds} Phase 2: {len(unmatched)} images → {n_new} new clusters")
        else:
            # Zero-shot: all images go to Phase 2
            all_indices = list(range(len(test_emb)))
            unknown_clusters = phase2_unknown_clustering(
                test_emb, all_indices, UNKNOWN_CLUSTER_DIST
            )
            info["matched"] = {}
            info["unknown_clusters"] = unknown_clusters
            n_new = len(set(unknown_clusters.values())) if unknown_clusters else 0
            print(f"\n{ds} (zero-shot): {len(all_indices)} images → {n_new} clusters")

    # Step 4: Build submission
    print(f"\n{'='*60}")
    print("Building submission...")
    submission = build_submission(data, metadata)

    # Ensure output directory exists
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved to {SUBMISSION_PATH}")

    # Step 5: Sanity checks
    sanity_check(submission, metadata)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
