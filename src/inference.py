import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_embeddings(model, dataloader, device='cpu'):
    model.eval()
    embeddings_dict = {}

    with torch.no_grad():
        for images, info in tqdm(dataloader, desc='Extracting embeddings'):
            embeddings = model(images.to(device)).cpu().numpy()
            for i, path in enumerate(info['image_path']):
                embeddings_dict[path] = embeddings[i]

    return embeddings_dict


def compute_similarity_matrix(query_embs, gallery_embs, metric='cosine'):
    if metric == 'cosine':
        q = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
        g = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-8)
        return np.matmul(q, g.T)
    elif metric == 'euclidean':
        return -np.linalg.norm(query_embs[:, None, :] - gallery_embs[None, :, :], axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def create_submission(query_ids, gallery_ids, similarity_matrix, output_path, top_k=100):
    predictions = []
    for i, qid in enumerate(tqdm(query_ids, desc='Creating submission')):
        top_idx = np.argsort(similarity_matrix[i])[::-1][:top_k]
        predictions.append(' '.join(gallery_ids[j] for j in top_idx))

    pd.DataFrame({'query_id': query_ids, 'predictions': predictions}).to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
