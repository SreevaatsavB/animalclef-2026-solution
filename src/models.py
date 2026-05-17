"""
Model definitions and utilities
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ReIDModel(nn.Module):
    """Base Re-Identification Model"""

    def __init__(self, backbone: str = 'resnet50', embedding_dim: int = 512, pretrained: bool = True):
        """
        Args:
            backbone: Backbone architecture ('resnet50', 'resnet101', etc.)
            embedding_dim: Dimension of the embedding vector
            pretrained: Use pretrained weights
        """
        super().__init__()

        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Embedding head
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

        # L2 normalization for embeddings
        self.normalize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]

        Returns:
            embeddings: Normalized embedding vectors [B, embedding_dim]
        """
        # Extract features
        features = self.backbone(x)

        # Get embeddings
        embeddings = self.embedding(features)

        # L2 normalize
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings


def load_megadescriptor_model(device='cpu'):
    """
    Load the MegaDescriptor foundation model from HuggingFace Hub

    Args:
        device: Device to load model on ('cpu', 'cuda', or 'mps')

    Returns:
        feature_extractor: DeepFeatures instance with MegaDescriptor model
    """
    try:
        import timm
        from wildlife_tools.features.deep import DeepFeatures

        # Load MegaDescriptor from HuggingFace Hub
        # Options: MegaDescriptor-L-384, MegaDescriptor-L-224, MegaDescriptor-B-224, etc.
        model_name = 'hf-hub:BVRA/MegaDescriptor-L-384'
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Create feature extractor
        feature_extractor = DeepFeatures(
            model=backbone,
            device=device,
            batch_size=32,
            num_workers=0
        )

        return feature_extractor
    except ImportError as e:
        raise ImportError(f"Required packages not installed: {e}\n"
                         "Install with: pip install wildlife-tools timm")
