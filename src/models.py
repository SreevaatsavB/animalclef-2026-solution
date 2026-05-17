import torch
import torch.nn as nn
from torchvision import models


class ReIDModel(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=512, pretrained=True):
        super().__init__()

        if backbone == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            base = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)


def load_megadescriptor_model(device='cpu'):
    import timm
    from wildlife_tools.features.deep import DeepFeatures

    # options: MegaDescriptor-T-224, MegaDescriptor-L-224, MegaDescriptor-L-384
    backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0)
    return DeepFeatures(model=backbone, device=device, batch_size=32, num_workers=0)
