from typing import List

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import torch
from models.semantic_vae import (
    DecoderBlock,
    DownSampleBlock,
    EncoderBlock,
    ResBlock,
    SemanticVAE,
)


@pytest.mark.parametrize(
    "in_channels,out_channels,bottleneck_ratio",
    [
        (4, 4, 1),
        (4, 6, 1),
        (6, 4, 1),
        (4, 4, 0.5),
        (4, 6, 0.5),
        (6, 4, 0.5),
        (4, 4, 2.0),
        (4, 6, 2.0),
        (6, 4, 2.0),
    ],
)
def test_resblock(
    in_channels: int, out_channels: int, bottleneck_ratio: int
):
    batch_size = 8
    height = width = 4

    block = ResBlock(in_channels, out_channels, bottleneck_ratio)

    input = torch.rand((batch_size, in_channels, height, width))
    expected_out_shape = (
        batch_size,
        out_channels,
        height,
        width,
    )

    out = block(input)

    assert expected_out_shape == out.shape, "Shape is incorrect"


@pytest.mark.parametrize(
    "in_channels,out_channels,downsample_factor",
    [
        (4, 4, 1),
        (4, 6, 1),
        (6, 4, 1),
        (4, 4, 2),
        (4, 6, 2),
        (6, 4, 2),
        (4, 4, 3),
        (4, 6, 3),
        (6, 4, 3),
    ],
)
def test_downsampleblock(
    in_channels: int, out_channels: int, downsample_factor: int
):
    batch_size = 8
    out_height = out_width = 4
    in_height = in_width = 4 * downsample_factor

    down_block = DownSampleBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        downsample_factor=downsample_factor,
    )

    input = torch.rand((batch_size, in_channels, in_height, in_width))
    expected_out_shape = (
        batch_size,
        out_channels,
        out_height,
        out_width,
    )

    out = down_block(input)

    assert expected_out_shape == out.shape


@pytest.mark.parametrize("in_channels", [1, 4])
@pytest.mark.parametrize("out_channels", [1, 4])
@pytest.mark.parametrize("bottleneck_ratio", [0.5, 1, 2])
@pytest.mark.parametrize("downsample_factor", [1, 2])
def test_encoder_block(
    in_channels: int,
    out_channels: int,
    bottleneck_ratio: float,
    downsample_factor: int,
):
    batch_size = 8
    out_height = out_width = 4
    in_height = in_width = 4 * downsample_factor

    block = EncoderBlock.make_block(
        in_channels, out_channels, bottleneck_ratio, downsample_factor
    )

    input = torch.rand((batch_size, in_channels, in_height, in_width))
    expected_out_shape = batch_size, out_channels, out_height, out_width
    expected_skip_shape = batch_size, out_channels, in_height, in_width

    out, skip = block(input)

    assert expected_out_shape == out.shape
    assert expected_skip_shape == skip.shape


@pytest.mark.parametrize("in_channels", [1, 4])
@pytest.mark.parametrize("out_channels", [1, 4])
@pytest.mark.parametrize("expansion", [1, 2, 4])
def test_decoder_make(
    in_channels: int,
    out_channels: int,
    expansion: int,
):
    batch_size = 4
    in_height = in_width = 4
    out_height = out_width = 4 * expansion
    in_c = skip_c = in_channels
    out_c = latent_c = out_channels
    block = DecoderBlock.make_block(
        in_c, skip_c, latent_c, out_c, expansion
    )

    input = torch.rand((batch_size, in_c, in_height, in_width))
    expected_out_shape = (batch_size, out_c, out_height, out_width)

    # Check if it works without the skip layer
    out = block(input)["out"]
    assert (
        expected_out_shape == out.shape
    ), "Shape is incorrect for normal input"

    # Check if it works with the skip layer
    input_skip = torch.rand((batch_size, skip_c, out_height, out_width))
    out = block(input, input_skip)["out"]
    assert (
        expected_out_shape == out.shape
    ), "Shape is incorrect for input with skip connections"


@pytest.mark.parametrize(
    "channels,reductions,bottlenecks",
    [
        ([4, 8], [2, 2], [0.5, 0.5]),
        ([4, 8], [4, 2], [0.5, 0.5]),
        ([4, 8], [16, 2], [0.5, 0.5]),
        ([4, 8], [8, 2], [0.5, 0.5]),
        ([4, 8, 16], [2, 2, 4], [1.0, 0.5, 0.5]),
        ([4, 8, 32], [2, 2, 8], [1.0, 0.5, 0.5]),
    ],
)
def test_semantic_vae_inference_shapes(
    test_image_batch,
    channels: List[int],
    reductions: List[int],
    bottlenecks: List[float],
    device: str,
):
    tot_reduction = np.prod(reductions)
    b, c, h, w = test_image_batch.shape
    num_classes = 8

    model = SemanticVAE(
        3, num_classes, channels, reductions, bottlenecks
    )
    test_image_batch = test_image_batch.to(device)
    model = model.to(device)

    z = model.encode_image(test_image_batch)[0]

    expected_latent_shape = torch.Size(
        [
            b,
            channels[-1],
            int(h / tot_reduction),
            int(w / tot_reduction),
        ]
    )
    assert (
        z.shape == expected_latent_shape
    ), f"Latent shape is not equal to expected: {z.shape} != {expected_latent_shape}"

    decoded = model.decode_image(z)["out"]

    assert (
        test_image_batch.shape == decoded.shape
    ), f"Shape of the decoded image is not equal. expected: {test_image_batch.shape} != {decoded.shape} :actual"

    segmentation = model.decode_label(z)["out"]

    expected_segmentation_shape = torch.Size([b, num_classes, h, w])
    assert (
        segmentation.shape == expected_segmentation_shape
    ), f"Segmentation shape is not equal to expected: {list(segmentation.shape)} != {expected_segmentation_shape}"


@pytest.mark.parametrize(
    "channels,reductions,bottlenecks",
    [
        ([4, 8], [2, 2], [0.5, 0.5]),
        ([4, 8], [4, 2], [0.5, 0.5]),
        ([4, 8], [16, 2], [0.5, 0.5]),
        ([4, 8], [8, 2], [0.5, 0.5]),
        ([4, 8, 16], [2, 2, 2], [1.0, 0.5, 0.5]),
        ([4, 8, 16], [2, 4, 1], [1.0, 0.5, 0.5]),
        ([4, 8, 32], [2, 8, 2], [1.0, 0.5, 0.5]),
        ([4, 8, 32], [4, 4, 2], [1.0, 0.5, 0.5]),
    ],
)
def test_semantic_vae_inference_shapes_e2e(
    test_image_batch,
    channels: List[int],
    reductions: List[int],
    bottlenecks: List[float],
    device: str,
):
    b, c, h, w = test_image_batch.shape
    num_classes = 8

    model = SemanticVAE(
        3, num_classes, channels, reductions, bottlenecks
    )
    test_image_batch = test_image_batch.to(device)
    model = model.to(device)

    expected_segmentation_shape = torch.Size([b, num_classes, h, w])
    segmentation = model(test_image_batch)["out"]
    assert (
        segmentation.shape == expected_segmentation_shape
    ), f"Segmentation shape is not equal to expected: {list(segmentation.shape)} != {expected_segmentation_shape}"
