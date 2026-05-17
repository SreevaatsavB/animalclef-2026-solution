"""
Inference and submission utilities
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for all images

    Args:
        model: The model to use for feature extraction
        dataloader: Data loader
        device: Device to run on

    Returns:
        embeddings_dict: Dictionary mapping image paths to embeddings
    """
    model.eval()
    embeddings_dict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeddings'):
            images, info = batch
            images = images.to(device)

            # Extract embeddings
            embeddings = model(images)
            embeddings = embeddings.cpu().numpy()

            # Store embeddings
            for i, img_path in enumerate(info['image_path']):
                embeddings_dict[img_path] = embeddings[i]

    return embeddings_dict


def compute_similarity_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute similarity matrix between query and gallery embeddings

    Args:
        query_embeddings: Query embeddings [N, D]
        gallery_embeddings: Gallery embeddings [M, D]
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        similarity_matrix: Similarity scores [N, M]
    """
    if metric == 'cosine':
        # Normalize embeddings
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity
        similarity_matrix = np.matmul(query_norm, gallery_norm.T)

    elif metric == 'euclidean':
        # Compute negative Euclidean distance (higher = more similar)
        similarity_matrix = -np.linalg.norm(
            query_embeddings[:, np.newaxis, :] - gallery_embeddings[np.newaxis, :, :],
            axis=2
        )
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return similarity_matrix


def create_submission(
    query_ids: List[str],
    gallery_ids: List[str],
    similarity_matrix: np.ndarray,
    output_path: str,
    top_k: int = 100
):
    """
    Create submission file for Kaggle

    Args:
        query_ids: List of query image IDs
        gallery_ids: List of gallery image IDs
        similarity_matrix: Similarity matrix [N_query, N_gallery]
        output_path: Path to save submission CSV
        top_k: Number of top matches to include
    """
    predictions = []

    for i, query_id in enumerate(tqdm(query_ids, desc='Creating submission')):
        # Get similarity scores for this query
        scores = similarity_matrix[i]

        # Get top-k most similar gallery images
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_gallery_ids = [gallery_ids[idx] for idx in top_indices]

        # Format as space-separated string
        predictions.append(' '.join(top_gallery_ids))

    # Create submission DataFrame
    submission = pd.DataFrame({
        'query_id': query_ids,
        'predictions': predictions
    })

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
