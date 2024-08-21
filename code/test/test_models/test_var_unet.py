from copy import deepcopy
import torch
from models import VariationalUNet


def test_forward_pass(image_batch):
    num_classes = 12
    model = VariationalUNet(
        image_channels=3,
        label_channels=num_classes,
        encoder_name="mobilenetv2_100",
        encoder_depth=5,
    )

    out = model(image_batch)
    assert "out" in out

    assert (
        out["out"].shape[0] == image_batch.shape[0]
    ), "Batch dimension is not equal"
    assert (
        out["out"].shape[2] == image_batch.shape[2]
    ), "Height dimension is not equal"
    assert (
        out["out"].shape[3] == image_batch.shape[3]
    ), "Width dimension is not equal"
    assert (
        out["out"].shape[1] == num_classes
    ), "Channel dimension is incorrect"


def test_saving_and_loading():
    in_c = 3
    batch_size = 8

    input = torch.rand((1, in_c, 32, 32)).expand(batch_size, -1, -1, -1)

    model = VariationalUNet(
        image_channels=3,
        label_channels=3,
        encoder_name="mobilenetv2_100",
    )
    # Ensure model is deterministic
    model.eval()

    pre_save_out = model(input)

    assert all(
        torch.equal(pre_save_out["out"][i - 1], pre_save_out["out"][i])
        for i in range(1, min(16, batch_size - 1))
    ), "Inference should be deterministic"

    model_state = deepcopy(model.state_dict())
    del model

    load_model = VariationalUNet(
        image_channels=3,
        label_channels=3,
        encoder_name="mobilenetv2_100",
        encoder_weights=None,
    )
    load_model.eval()

    pre_load_out = load_model(input)

    assert not torch.equal(pre_load_out["out"], pre_save_out["out"])

    load_model.load_state_dict(model_state)
    assert torch.equal(pre_save_out["out"], load_model(input)["out"])


def test_loading_part():
    in_c = 3
    batch_size = 8

    input = torch.rand((1, in_c, 32, 32)).expand(batch_size, -1, -1, -1)

    model = VariationalUNet(
        image_channels=3,
        label_channels=3,
        encoder_name="mobilenetv2_100",
        encoder_weights=None,
    )
    # Ensure model is deterministic
    model.eval()

    pre_save_dec = model(input)

    state_dict = deepcopy(model.state_dict())
    del model

    model = VariationalUNet(
        image_channels=3,
        label_channels=4,  # Different on purpose
        encoder_name="mobilenetv2_100",
        state_dict=state_dict,
        load_segmentation_head=False,
    )
    model.eval()

    post_save_dec = model(input)
    assert (
        post_save_dec["out"].size(1) == 4
    ), "Loaded model did not have the correct label channels"
    assert not torch.equal(
        pre_save_dec["out"], post_save_dec["out"]
    ), "Decoded result should be different"
