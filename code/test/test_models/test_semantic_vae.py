from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import torch
from models.semantic_vae import DecoderBlock, EncoderBlock, ResBlock, SemanticVAE


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
    ]
)
def test_resblock_make(in_channels: int, out_channels: int, reduction: int):
    batch_size = 16
    out_height = out_width = 4
    in_height = in_width = 4 * reduction
    block = ResBlock.make_block(in_channels, out_channels, reduction)
    
    input = torch.rand((batch_size, in_channels, in_height, in_width))
    expected_out_shape = (batch_size, out_channels, out_height, out_width)
    
    out = block(input)
    
    assert expected_out_shape == out.shape

@pytest.mark.parametrize(
    "channels,reduction",
    [
        ([4, 16], 4),
        ([4, 8, 16], 4),
    ]
)
def test_encoder_block(channels: List[int], reduction: int):
    batch_size = 16
    out_height = out_width = 4
    in_height = in_width = 4 * reduction
    
    block = EncoderBlock(channels, reduction)
    
    input = torch.rand((batch_size, channels[0], in_height, in_width))
    expected_out_shape = (batch_size, channels[-1], out_height, out_width)

    out = block(input)

    assert expected_out_shape == out.shape

@given(st.integers(1, 32), st.integers(1, 32), st.integers(1, 32), st.integers(1, 32), st.integers(1, 4))
@settings(deadline=None)
def test_decoder_make(in_channels: int, skip_channels: int, latent_channels: int, out_channels: int, expansion: int):
    batch_size = 16
    in_height = in_width = 4
    out_height = out_width = 4 * expansion
    in_channels=in_channels
    skip_channels=in_channels
    latent_channels=out_channels
    out_channels=out_channels
    block = DecoderBlock.make_block(in_channels, skip_channels, latent_channels, out_channels, expansion)

    input = torch.rand((batch_size, in_channels, in_height, in_width))
    expected_out_shape = (batch_size, out_channels, out_height, out_width)

    # Check if it works without the skip layer
    out = block(input)
    assert expected_out_shape == out.shape
    
    # Check if it works with the skip layer
    input_skip = torch.rand((batch_size, skip_channels, out_height, out_width))
    out = block(input, input_skip)
    assert expected_out_shape == out.shape

def test_semantic_vae_inference_shapes(test_image_batch):
    model = SemanticVAE()
    z = model.encode_image(test_image_batch)
    decoded = model.decode_image(z)
    semantic = model.decode_label(z)

    assert test_image_batch.shape == decoded.shape, "Shape of the decoded image is not equal"
