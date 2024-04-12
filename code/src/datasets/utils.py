from torchvision.tv_tensors import TVTensor


class LatentTensor(TVTensor):
    """Ensure latent tensors are not changed in transforms

    https://pytorch.org/vision/0.16/auto_examples/transforms/plot_custom_tv_tensors.html#sphx-glr-auto-examples-transforms-plot-custom-tv-tensors-py
    TODO: Determine if latent variables should have some implementations
        e.g.:
            hflip/vflip
    """

    pass
