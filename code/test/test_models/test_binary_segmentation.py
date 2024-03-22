import torch

from models.binary_segmentation_model import BinarySegmentationModel


def test_binary_segmentation_model(test_image_batch: torch.Tensor):
    BSModel = BinarySegmentationModel()
    mask = BSModel(test_image_batch)
    assert mask.shape[0] == test_image_batch.shape[0], "Invalid batch dimension size"
    assert mask.shape[1] == 1, "Invalid mask dimension size"
    assert mask.shape[2:] == test_image_batch.shape[2:], "Invalid WxH shape"


def test_save_load(test_image_batch: torch.Tensor):
    model = BinarySegmentationModel()
    pre_save = model(test_image_batch)
    model.save()