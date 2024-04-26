from models import MobileVAE


def test_forward_pass(image_batch):
    num_classes = 12
    model = MobileVAE(
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
