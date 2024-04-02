from typing import List

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import torch
from models.semantic_vae import (
    DecoderBlock,
    EncoderBlock,
    ResBlock,
    SemanticVAE,
)


@pytest.mark.parametrize(
    "in_channels,out_channels,reduction",
    [
        (4, 4, 16),
        (4, 4, 4),
        (4, 4, 2),
        (4, 4, 1),
        (4, 8, 16),
        (4, 8, 4),
        (4, 8, 2),
        (4, 8, 1),
        (8, 4, 16),
        (8, 4, 4),
        (8, 4, 2),
        (8, 4, 1),
    ],
)
def test_resblock_make(in_channels: int, out_channels: int, reduction: int):
    batch_size = 16
    out_height = out_width = 4
    in_height = in_width = 4 * reduction
    block = ResBlock.make_block(in_channels, out_channels, reduction)

    input = torch.rand((batch_size, in_channels, in_height, in_width))
    expected_out_shape = (
        batch_size,
        out_channels,
        out_height,
        out_width,
    )

    out = block(input)

    assert expected_out_shape == out.shape


@pytest.mark.parametrize(
    "channels,reduction",
    [
        ([4, 16], 4),
        ([4, 8, 16], 4),
    ],
)
def test_encoder_block(channels: List[int], reduction: int):
    batch_size = 16
    out_height = out_width = 4
    in_height = in_width = 4 * reduction

    block = EncoderBlock(channels, reduction)

    input = torch.rand((batch_size, channels[0], in_height, in_width))
    expected_out_shape = (
        batch_size,
        channels[-1],
        out_height,
        out_width,
    )

    out = block(input)

    assert expected_out_shape == out.shape


@given(
    st.integers(1, 4),
    st.integers(1, 4),
    st.integers(1, 4),
    st.integers(1, 4),
    st.integers(1, 4),
)
@settings(deadline=None)
def test_decoder_make(
    in_channels: int,
    skip_c: int,
    latent_c: int,
    out_channels: int,
    expansion: int,
):
    batch_size = 4
    in_height = in_width = 4
    out_height = out_width = 4 * expansion
    in_c = skip_c = in_channels
    out_c = latent_c = out_channels
    block = DecoderBlock.make_block(in_c, skip_c, latent_c, out_c, expansion)

    input = torch.rand((batch_size, in_c, in_height, in_width))
    expected_out_shape = (batch_size, out_c, out_height, out_width)

    # Check if it works without the skip layer
    out = block(input)
    assert expected_out_shape == out.shape

    # Check if it works with the skip layer
    input_skip = torch.rand((batch_size, skip_c, out_height, out_width))
    out = block(input, input_skip)
    assert expected_out_shape == out.shape


@pytest.mark.parametrize(
    "channels,reductions",
    [
        ([4, 16], [2, 2]),
        ([4, 16, 32], [2, 2, 2]),
        ([4, 16, 64], [2, 2, 2]),
    ],
)
def test_semantic_vae_inference_shapes(
    test_image_batch, channels: List[int], reductions: List[int]
):
    tot_reduction = np.prod(reductions)
    b, c, h, w = test_image_batch.shape
    num_classes = 8

    model = SemanticVAE(
        3,
        num_classes,
        channels,
        reductions,
    )

    z = model.encode_image(test_image_batch)

    expected_latent_shape = torch.Size(
        [b, channels[-1], int(h / tot_reduction), int(w / tot_reduction)]
    )
    assert (
        z.shape == expected_latent_shape
    ), f"Latent shape is not equal to expected: {z.shape} != {expected_latent_shape}"

    decoded = model.decode_image(z)

    assert (
        test_image_batch.shape == decoded.shape
    ), f"Shape of the decoded image is not equal. expected: {test_image_batch.shape} != {decoded.shape} :actual"

    segmentation = model.decode_label(z)

    expected_segmentation_shape = torch.Size([b, num_classes, h, w])
    assert (
        segmentation.shape == expected_segmentation_shape
    ), f"Segmentation shape is not equal to expected: {list(segmentation.shape)} != {expected_segmentation_shape}"
