"""Determinestic Uncertainty Quantification (DUQ)

This is a head based on DUQ [1], adapted to work 
as a convolutional head.

[1] Van Amersfoort, J., Smith, L., Teh, Y. W., & Gal, Y. (2020, November).
    Uncertainty estimation using a single deep deterministic neural network.
    In International conference on machine learning (pp. 9690-9700). PMLR.
    https://arxiv.org/abs/2003.02037
"""

from typing import Optional, Tuple
import torch
from torch import nn


class DUQHead(nn.Module):
    """Deterministic Uncertainty Quantification Head

    Args:
        in_channels: int            The number of output channels of the feature extractor.
        num_classes: int            The amount of output classes.
        embedding_size: int [8]     The internal embedding size used for the centroids.
        gamma: float        [0.99]  Hyperparameter for the exponential moving average (EMA) of the centroids.
        length_scale: float [0.10]  Hyperparameter for the length_scale of the RBF kernel.

        # The following arguments are intendent for testing purposes only
        _m: Optional[torch.Tensor]          Set the initial values for the m accumulator, used to calculate EMA centroids.
        _N: Optional[torch.Tensor]          Set the initial values for the N accumulator, used to calculate EMA centroids.
        _weights: Optional[torch.Tensor]    Set the initial weights.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embedding_size: int = 8,
        gamma: float = 0.99,
        length_scale: float = 0.1,
        _m: Optional[torch.Tensor] = None,
        _N: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.weights = nn.Parameter(
            torch.normal(0, 0.05, (embedding_size, num_classes, in_channels))
        )

        # Sigma in the paper
        self.register_buffer("length_scale", torch.tensor([length_scale]))

        self.register_buffer("gamma", torch.tensor([gamma]))

        # m keeps track of the moving average sum of centroids for that class
        # N keeps track of the moving average frequency of centroids for that class
        if _m is not None:
            assert _m.shape == (
                embedding_size,
                num_classes,
            ), "The given `_m` is not the expected shape"
        else:
            _m = torch.normal(0, 1, (embedding_size, num_classes))
        self.register_buffer("m", _m)

        if _N is not None:
            assert _N.shape == (num_classes), "The given `_N` is not the expected shape"
        else:
            _N = torch.ones(num_classes)
        # 12 is hardcoded in the original implementation, not sure why.
        # A possible reason could be that this is the BS / number of classes in their examples
        # If that is the case, for us it should be the 1 / num_classes, as our centroids
        # are BS independent.
        self.register_buffer("N", torch.ones(num_classes) * 12)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = self.calculate_embeddings(features)
        logits = _rbf(embeddings, self.centroids, self.length_scale)
        return logits
    
    def calculate_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        return _conv_duq_last_layer(features, self.weights)

    @property
    def centroids(self) -> torch.Tensor:
        return self.m / self.N

    def update_centroids(self, features: torch.Tensor, y_true: torch.Tensor):
        self.N, self.m = _update_centroids(
            self.N, self.m, self.gamma, self.calculate_embeddings(features), y_true
        )


def _conv_duq_last_layer(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Forward pass of the 'last_layer' from duq.

    Calculates the embedding of each pixel which is later compared
    with the average centroids of the training dataset of each class.

    This implementation is quite slow, it might be required to
    implement it as a 1x1 conv later, not sure if that is faster.
    """
    return torch.einsum("...j,mnj->...mn", features, weights)


def _rbf(
    embeddings: torch.Tensor, centroids: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Calculate the RBF kernel between embeddings and centroids

    maximal iff embeddings == centroids -> _rbf(embeddings, centroids) == torch.ones_like(embeddings)
    """
    gamma = -1 / (2 * sigma**2)
    return (embeddings - centroids.unsqueeze(0)).pow(2).mean(-2).mul(gamma).exp()


def _update_centroids(
    prev_N: torch.Tensor,
    prev_m: torch.Tensor,
    gamma: float,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = labels.shape[0]

    batch_frequency = labels.sum(0) / bs
    new_N = gamma * prev_N + (1 - gamma) * batch_frequency
    embedding_sum = torch.einsum("ijk,ik->jk", embeddings, labels)
    embedding_sum = embedding_sum / bs
    new_m = gamma * prev_m + (1 - gamma) * embedding_sum

    return new_N, new_m
