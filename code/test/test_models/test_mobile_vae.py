from copy import deepcopy
import torch
from models import MobileVAE


def test_forward_pass(image_batch):
    num_classes = 12
    model = MobileVAE(
        label_channels=num_classes,
        encoder_name="mobilenetv2_100",
        encoder_depth=1,
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

    model = MobileVAE(
        label_channels=3,
        encoder_name="mobilenetv2_100",
        encoder_depth=3,
    )
    # Ensure model is deterministic
    model.eval()

    pre_save_enc = model.encode(input)
    pre_save_out = model(input)

    assert all(
        torch.equal(pre_save_out["out"][i - 1], pre_save_out["out"][i])
        for i in range(1, min(16, batch_size - 1))
    ), "Inference should be deterministic"

    model_state = deepcopy(model.state_dict())
    del model

    load_model = MobileVAE(
        label_channels=3,
        encoder_name="mobilenetv2_100",
        # encoder_weights=None,
        encoder_depth=3,
    )
    load_model.eval()

    pre_load_out = load_model(input)

    assert not torch.equal(pre_load_out["out"], pre_save_out["out"])

    load_model.load_state_dict(model_state)
    post_load_enc = load_model.encode(input)

    assert torch.equal(pre_save_out["out"], load_model(input)["out"])

    assert torch.equal(
        pre_save_enc[0]["out"], post_load_enc[0]["out"]
    ), "Midblock did not have same result"
    assert all(
        torch.equal(pre, post)
        for (pre, post) in zip(pre_save_enc[1:], post_load_enc[1:])
    ), "Encoding should be equal"


def test_loading_part():
    in_c = 3
    batch_size = 8

    input = torch.rand((1, in_c, 32, 32)).expand(batch_size, -1, -1, -1)

    model = MobileVAE(
        label_channels=3,
        encoder_name="mobilenetv2_100",
        encoder_weights=None,
        encoder_depth=3,
    )
    # Ensure model is deterministic
    model.eval()

    pre_save_enc = model.encode(input)
    pre_save_dec = model(input)

    state_dict = deepcopy(model.state_dict())
    del model

    model = MobileVAE(
        label_channels=4,  # Different on purpose
        encoder_name="mobilenetv2_100",
        encoder_depth=3,
        state_dict=state_dict,
        load_decoder=False,
    )
    model.eval()

    post_save_enc = model.encode(input)
    assert torch.equal(
        pre_save_enc[0]["out"], post_save_enc[0]["out"]
    ), "Midblock did not have same result"
    assert all(
        torch.equal(pre, post)
        for (pre, post) in zip(pre_save_enc[1:], post_save_enc[1:])
    ), "Encoding should be equal"

    post_save_dec = model(input)
    assert (
        post_save_dec["out"].size(1) == 4
    ), "Loaded model did not have the correct label channels"
    assert not torch.equal(
        pre_save_dec["out"], post_save_dec["out"]
    ), "Decoded result should be different"
