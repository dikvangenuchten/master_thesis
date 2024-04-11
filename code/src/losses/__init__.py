from .duq_loss import DUQLoss
from .gradient_penalty import GradientPenalty
from .kl_divergence import KLDivergence
from .weighted_loss import WeightedLoss
from .wrapper_loss import WrappedLoss

__all__ = [
    "DUQLoss",
    "GradientPenalty",
    "KLDivergence",
    "WeightedLoss",
    "WrappedLoss",
]
