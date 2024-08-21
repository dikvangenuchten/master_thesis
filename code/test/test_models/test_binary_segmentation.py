import torch

from models.binary_segmentation_model import BinarySegmentationModel


def test_binary_segmentation_model(image_batch: torch.Tensor):
    BSModel = BinarySegmentationModel()
    mask = BSModel(image_batch)["out"]
    assert (
        mask.shape[0] == image_batch.shape[0]
    ), "Invalid batch dimension size"
    assert mask.shape[1] == 1, "Invalid mask dimension size"
    assert mask.shape[2:] == image_batch.shape[2:], "Invalid WxH shape"


def test_save_load(image_batch: torch.Tensor, tmp_path: str):
    model = BinarySegmentationModel()

    path = tmp_path / "model.pt"
    pre_save = model(image_batch)["out"]
    torch.save(model, path)
    loaded = torch.load(path)
    post_load = loaded(image_batch)["out"]

    assert torch.allclose(pre_save, post_load)
