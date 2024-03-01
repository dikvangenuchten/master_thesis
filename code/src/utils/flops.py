from torchtnt.utils.flops import FlopTensorDispatchMode
import torch
from torch import nn


@torch.no_grad
def calc_flops_per_pixel(model: nn.Module, verbose: bool = False) -> torch.Tensor:
    model = model.to("cuda")
    input = torch.rand(size=(1, 3, 64, 64), device="cuda")
    with FlopTensorDispatchMode(model) as ftdm:
        _ = model(input)
        flop_counts64 = ftdm.flop_counts
        del _

    if verbose:
        input = torch.rand(size=(1, 3, 96, 96), device="cuda")
        with FlopTensorDispatchMode(model) as ftdm:
            _ = model(input)
            flop_counts96 = ftdm.flop_counts
            del _

        input = torch.rand(size=(4, 3, 64, 64), device="cuda")
        with FlopTensorDispatchMode(model) as ftdm:
            _ = model(input)
            flop_counts64b = ftdm.flop_counts
            del _

        input = torch.rand(size=(4, 3, 96, 96), device="cuda")
        with FlopTensorDispatchMode(model) as ftdm:
            _ = model(input)
            flop_counts96b = ftdm.flop_counts
            del _

        flop_counts64_i = 1 * 3 * 64 * 64
        flop_counts96_i = 1 * 3 * 96 * 96
        flop_counts64b_i = 4 * 3 * 64 * 64
        flop_counts96b_i = 4 * 3 * 96 * 96

        print("n_input, flop_counts, conv_flop normalized, bmm_flop normalized")
        print(
            f"{flop_counts64_i}\t {flop_counts64_i / flop_counts64_i} \t {dict(flop_counts64[''])}\t {flop_counts64['']['convolution.default'] / flop_counts64['']['convolution.default']}\t {flop_counts64['']['bmm.default'] / flop_counts64['']['bmm.default']}"
        )
        print(
            f"{flop_counts96_i}\t {flop_counts96_i / flop_counts64_i} \t {dict(flop_counts96[''])}\t {flop_counts96['']['convolution.default'] / flop_counts64['']['convolution.default']}\t {flop_counts96['']['bmm.default'] / flop_counts64['']['bmm.default']}"
        )
        print(
            f"{flop_counts64b_i}\t {flop_counts64b_i / flop_counts64_i} \t {dict(flop_counts64b[''])}\t {flop_counts64b['']['convolution.default'] / flop_counts64['']['convolution.default']}\t {flop_counts64b['']['bmm.default'] / flop_counts64['']['bmm.default']}"
        )
        print(
            f"{flop_counts96b_i}\t {flop_counts96b_i / flop_counts64_i} \t {dict(flop_counts96b[''])}\t {flop_counts96b['']['convolution.default'] / flop_counts64['']['convolution.default']}\t {flop_counts96b['']['bmm.default'] / flop_counts64['']['bmm.default']}"
        )
    return (
        flop_counts64[""]["convolution.default"] + flop_counts64[""]["bmm.default"]
    ) / (64 * 64)
