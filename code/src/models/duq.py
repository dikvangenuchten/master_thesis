"""Determinestic Uncertainty Quantification (DUQ)

This is a head based on DUQ [1], adapted to work 
as a convolutional head.

[1] Van Amersfoort, J., Smith, L., Teh, Y. W., & Gal, Y. (2020, November).
    Uncertainty estimation using a single deep deterministic neural network.
    In International conference on machine learning (pp. 9690-9700). PMLR.
    https://arxiv.org/abs/2003.02037
"""

from torch import nn


class DUQHead(nn.Module):
    """Deterministic Uncertainty Quantification Head

    Args:
        num_classes: int        The amount of output classes
        embedding_size: int     The internal embedding size used for the centroids
        gamma: float            Hyperparameter for the moving average of the centroids

    """

    def __init__(self, num_classes: int, embedding_size: int, gamma: float = 0.99):
        pass

    def forward(self, features):
        pass

    def predict(self):
        pass
