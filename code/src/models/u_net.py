import segmentation_models_pytorch as smp
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert (
            in_channels == 3
        ), "Currently only RGB images are supported (due to pretrained weights)"
        self.backbone = smp.Unet(
            encoder_name="timm-mobilenetv3_small_075",
            encoder_depth=5,
            encoder_weights="imagenet",
            classes=out_channels,
            activation=None,
        )

        # TODO add DUQ/Evidence Based Head
        # self.semantic_duq_head

    def forward(self, img):
        x = self.backbone(img)
        return x
