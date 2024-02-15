"""Determinestic Uncertainty Quantification (DUQ)

This is a head based on DUQ [1], adapted to work 
as a convolutional head.

[1] Van Amersfoort, J., Smith, L., Teh, Y. W., & Gal, Y. (2020, November).
    Uncertainty estimation using a single deep deterministic neural network.
    In International conference on machine learning (pp. 9690-9700). PMLR.
    https://arxiv.org/abs/2003.02037
"""

import torch
from torch import nn


class DUQHead(nn.Module):
    """Deterministic Uncertainty Quantification Head

    Args:
        num_classes: int        The amount of output classes
        embedding_size: int     The internal embedding size used for the centroids
        gamma: float            Hyperparameter for the moving average of the centroids

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embedding_size: int = 8,
        gamma: float = 0.99,
        length_scale: float = 0.1,
    ):
        self.weights = nn.Parameter(
            torch.normal((embedding_size, num_classes, in_channels))
        )
        # Sigma in the paper
        self.length_scale = self.register_buffer(
            "length_scale", torch.tensor([length_scale])
        )

        self.m = self.register_buffer("m", torch.normal(0, 1, (embedding_size, num_classes)))
        # 12 is hardcoded in the original implementation, not sure why.
        self.N = self.register_buffer("N", torch.ones(num_classes) * 12)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = _conv_duq_last_layer(features, self.weights)
        
        raise NotImplementedError("#TODO calculate the centroids and distance")

    def update_centroids(self, features: torch.Tensor, y_true: torch.Tensor):
        pass


def _conv_duq_last_layer(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Forward pass of the 'last_layer' from duq"""
    return torch.einsum("...j,mnj->...mn", features, weights)


def _distance(embeddings: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    (embeddings - centroids)
    pass