"""

Reference implementations are based on the original papers Github repository:
https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/cnn_duq.py
"""

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from models.duq import DUQHead, _conv_duq_last_layer, _rbf, _update_centroids


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
    """Apply the reference implementation to each value in the feature matrix in a conv-like (1x1) manner"""
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


def reference_rbf_implementation(embeddings, centroids, sigma):
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
        Furthermore, it uses `z` for the model output (we use embeddings).
    """
    diff = embeddings - centroids.unsqueeze(0)
    distances = (-(diff**2)).mean(1).div(2 * sigma**2).exp()
    distances_2 = (diff**2).mean(1).div(2 * sigma**2).mul(-1).exp()
    assert (distances == distances_2).all()
    return distances


def reference_rbf_implementation_conv(embeddings, centroids, sigma):
    b, h, w, m, c = embeddings.shape
    m_, c_ = centroids.shape
    assert m == m_, "Embedding dimensions are not equal"
    assert c == c_, "Class dimensions are not equal"

    output = torch.zeros(b, h, w, c, device=embeddings.device)
    for i in range(h):
        for j in range(w):
            output[:, i, j] = reference_rbf_implementation(
                embeddings[:, i, j], centroids, sigma
            )
    return output


def test_reference_distance_implementation():
    batch_size = 16
    embedding_size = 4
    num_classes = 8

    centroids = torch.rand((embedding_size, num_classes))
    embeddings = centroids.expand((batch_size, embedding_size, num_classes))

    assert (reference_rbf_implementation(embeddings, centroids, 0.1) == 1).all()
    assert (reference_rbf_implementation(embeddings, centroids + 0.001, 0.1) < 1).all()


@pytest.mark.parametrize("sigma", [0.01, 0.1, 0.5, 1.0, 2.0])
def test_conv_distance_implementation(sigma: float):
    batch_size = 16
    embedding_size = 4
    height, width = 6, 12
    num_classes = 8

    centroids = torch.rand((embedding_size, num_classes))
    embeddings = torch.rand((batch_size, height, width, embedding_size, num_classes))

    reference = reference_rbf_implementation_conv(embeddings, centroids, sigma)
    assert tuple(reference.shape) == (
        batch_size,
        height,
        width,
        num_classes,
    ), "reference did not have the expected shape"

    own = _rbf(embeddings, centroids, sigma)
    assert tuple(own.shape) == (
        batch_size,
        height,
        width,
        num_classes,
    ), "own did not have the expected shape"

    assert torch.allclose(
        reference, own
    ), "Reference and Own implementation are not equal"


def test_forward_pass_shape():
    batch_size = 1
    height = 2
    width = 3

    feature_size = 8
    num_classes = 6
    embedding_size = 4

    duq_layer = DUQHead(
        in_channels=feature_size, num_classes=num_classes, embedding_size=embedding_size
    )

    input = torch.rand((batch_size, height, width, feature_size))
    output = duq_layer(input)
    assert output.shape == (batch_size, height, width, num_classes)


def test_forward_pass_batch_indepent():
    """The output of each sample should be independent of the other samples"""
    batch_size = 16
    height = 3
    width = 3

    feature_size = 8
    num_classes = 4
    embedding_size = 2

    duq_layer = DUQHead(
        in_channels=feature_size, num_classes=num_classes, embedding_size=embedding_size
    )

    input = torch.rand((1, height, width, feature_size))
    expected = duq_layer(input)

    rand_batch = torch.rand((batch_size, height, width, feature_size))
    for i in range(batch_size):
        in_ = rand_batch.clone()
        in_[i] = input

        out = duq_layer(in_)

        assert torch.allclose(out[i], expected[0])


def reference_implementation_update_centroids(
    N: torch.Tensor,
    m: torch.Tensor,
    gamma: float,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
):
    """Reference implementation of the update centroid.

    Args:
        N (torch.Tensor) [C]: The EMA of the frequency per class
        m (torch.Tensor) [E, C]: The EMA of the embeddings per class
        gamma (float): Hyperparameter for EMA \in [0, 1]
        embeddings (torch.Tensor) [B, E, C]: Model embedding output
        labels (torch.Tensor) [B, C]: One hot encoded labels

    Returns:
        N', m' (Tuple[torch.Tensor, torch.Tensor]): _description_
    """
    # Check shapes
    assert N.ndim == 1, f"N should be 1d not {N.ndim}d. ([{N.shape}])"
    assert m.ndim == 2, f"m should be 2d not {m.ndim}d. ([{m.shape}])"
    assert embeddings.ndim == 3, f"embeddings should be 3d not {embeddings.ndim}d"
    assert labels.ndim == 2, f"labels should be 2d (one-hot encoded) not {labels.ndim}d"

    batch_size, embedding_size, num_classes = embeddings.shape
    assert (
        labels.shape[0] == batch_size
    ), f"Labels batch dim ({labels.shape[0]}) does not match embeddings batch dim ({batch_size})"

    assert (
        N.shape[0] == num_classes
    ), f"Ambiguos number of classes: {N.shape[0]} (N) != {num_classes} (embeddings)"
    assert (
        m.shape[1] == num_classes
    ), f"Ambiguos number of classes: {m.shape[1]} (m) != {num_classes} (embeddings)"
    assert (
        labels.shape[-1] == num_classes
    ), f"Ambiguos number of classes: {labels.shape[-1]} (labels) != {num_classes} (embeddings)"

    assert (
        m.shape[0] == embedding_size
    ), f"Ambiguos embedding size: {m.shape[0]} (m) != {embedding_size} (embeddings)"

    new_N = gamma * N + (1 - gamma) * labels.sum(0)
    embedding_sum = torch.einsum("ijk,ik->jk", embeddings, labels)
    new_m = gamma * m + (1 - gamma) * embedding_sum
    return new_N, new_m


def test_reference_update_centroid():
    batch_size = 3
    num_classes = 3
    embedding_size = 2
    gamma = 0.99

    # Start with an initial m of 0 and a N of 1/3
    m = torch.zeros((embedding_size, num_classes))
    N = torch.ones((num_classes)) / 3

    # labels = torch.eye(batch_size)
    labels = torch.eye(batch_size, dtype=torch.long)

    embeddings = torch.tensor(
        [
            [[-1, 0, 1], [-1, 0, 1]],
            [[-1, 0, 1], [-1, 0, 1]],
            [[-1, 0, 1], [-1, 0, 1]],
        ]
    )

    ev = 1 - gamma
    expected_m = torch.tensor([[-ev, 0, ev], [-ev, 0, ev]])

    act_N, act_m = reference_implementation_update_centroids(
        N, m, gamma, embeddings, labels
    )

    assert torch.allclose(act_m, expected_m)


@pytest.mark.parametrize("gamma", np.linspace(1, 0, 10, endpoint=False))
def test_update_centroid_against_reference(gamma: float):
    batch_size = 4
    embedding_size = 8
    num_classes = batch_size * 16

    labels = F.one_hot(torch.randint(0, num_classes, (batch_size,)), num_classes).to(
        dtype=torch.float32
    )

    embeddings = torch.rand((batch_size, embedding_size, num_classes))

    N = torch.rand((num_classes))
    m = torch.rand((embedding_size, num_classes))

    ref_N, ref_m = reference_implementation_update_centroids(
        N, m, gamma, embeddings, labels
    )

    # We use a batch independent version of EMA, hence the initial N and m need to be divided by the batch_size
    #
    # Using the average per batch, instead of the sum per batch has as benefit that the value N and m are independent of the batch_size, and can more easily be updated after the fact
    own_N, own_m = _update_centroids(
        N / batch_size, m / batch_size, gamma, embeddings, labels
    )

    # Same reason as above, divide the reference N by the BS
    assert torch.allclose(ref_N / batch_size, own_N), "Count is not updated correctly"
    assert torch.allclose(ref_m / batch_size, own_m), "Sum is not updated correctly"

    # The resulting centroids should be the same (i.e. m / N is BS independent for both)
    assert torch.allclose(
        ref_m / ref_N, own_m / own_N
    ), "Centroids are not updated correctly"


def test_update_centroid_indepent():
    """Each centroid is independent of each other centroid"""
    feature_size = 8
    num_classes = 4
    embedding_size = 2

    duq_layer = DUQHead(
        in_channels=feature_size, num_classes=num_classes, embedding_size=embedding_size
    )

    # Get initial centroids
    centroids = duq_layer.centroids

    # Given a batch which does not

    # Given a gamma of 0, the moving average should be equal to the last value

    # Given a gamma of 1, the moving average should not update

    for i in range(num_classes):
        pass
