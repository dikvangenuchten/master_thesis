"""

Reference implementations are based on the original papers Github repository:
https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/cnn_duq.py
"""

import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from models.duq import _conv_duq_last_layer

def reference_duq_last_layer(feature_matrix, weight_matrix) -> torch.Tensor:
    """"""
    # Equal to y[i] = weights.matmul(feature_matrix[i])
    return torch.einsum("ij,mnj->imn", feature_matrix, weight_matrix)

@given(st.integers(1, 32), st.integers(1, 32), st.integers(1, 32), st.integers(1, 32))
def test_native_implementation(
    batch_size: int,
    num_features: int,
    embedding_size: int,
    num_classes: int,
):
    feature_matrix = torch.randint(-256, 256, (batch_size, num_features))
    weight_matrix = torch.randint(
        -256, 256, (embedding_size, num_classes, num_features)
    )

    reference = reference_duq_last_layer(feature_matrix, weight_matrix)
    implementation = _conv_duq_last_layer(feature_matrix, weight_matrix)

    assert (
        reference == implementation
    ).all(), f"Reference and implementation are not equal for {feature_matrix =}, {weight_matrix =}"


def reference_duq_conv(feature_matrix, weight_matrix) -> torch.Tensor:
    """Apply the reference implementation to each value in the feature matrix in a conv-like manner"""
    b, h, w, f = feature_matrix.shape
    e, c, f_ = weight_matrix.shape
    assert f == f_
    output = torch.zeros(b, h, w, e, c, device=feature_matrix.device)
    for i in range(h):
        for j in range(w):
            output[:, i, j] = reference_duq_last_layer(
                feature_matrix[:, i, j], weight_matrix
            )
    return output

# Comprehensive test
# Takes ~4 minutes
# import pytest
#
#
# @pytest.mark.parametrize("batch_size", [1, 2, 16, 32], ids="b({})".format)
# @pytest.mark.parametrize("num_features", [1, 2, 16, 32], ids="f({})".format)
# @pytest.mark.parametrize("embedding_size", [1, 2, 16], ids="e({})".format)
# @pytest.mark.parametrize("num_classes", [1, 2, 16, 32], ids="c({})".format)
# @pytest.mark.parametrize("height", [1, 2, 128], ids="h({})".format)
# @pytest.mark.parametrize("width", [1, 2, 128], ids="w({})".format)


# Faster test
@given(
    st.integers(1, 8),
    st.integers(1, 8),
    st.integers(1, 8),
    st.integers(1, 8),
    st.integers(16, 128),
    st.integers(16, 128),
)
@settings(deadline=None, max_examples=10)
def test_conv_implementation_of_last_layer(
    batch_size: int,
    num_features: int,
    embedding_size: int,
    num_classes: int,
    height: int,
    width: int,
):
    # Cuda does not support integer tensors for this
    feature_matrix = torch.rand(
        (batch_size, height, width, num_features), device="cuda"
    )
    weight_matrix = torch.rand(
        (embedding_size, num_classes, num_features), device="cuda"
    )

    reference = reference_duq_conv(feature_matrix, weight_matrix).detach()
    implementation = _conv_duq_last_layer(feature_matrix, weight_matrix).detach()

    assert torch.allclose(
        reference, implementation
    ), f"Reference and Implementation are not equal for {feature_matrix =}, {weight_matrix =}"


def reference_distance_implementation(embeddings, centroids, sigma):
    """Taken from https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/
    
    There are 2 implementations in the codebase
    
    resnet_duq.py:  `(diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()`
    cnn_duq.py:     `(-(diff**2)).mean(1).div(2 * self.sigma**2).exp()`

    Args:
        embeddings (torch.Tensor): The output of the model
        centroids (torch.Tensor): The tracked average output of the model for each class (already normalized)
        
        sigma (float): Hyperparameter, also calles length_scale
    
    Note:
        The reference implementation calls the tracked average outputs embeddings (we use centroids)
        Furthermore, it uses `z` for the model output (we use embeddings)
    """
    diff = embeddings - centroids.unsqueeze(0)
    distances = (-(diff**2)).mean(1).div(2 * sigma**2).exp()
    return distances
    
@pytest.mark.parametrize("sigma", [0.1])
def test_conv_distance_implementation(sigma: float):
    assert False, "Continue here :)"