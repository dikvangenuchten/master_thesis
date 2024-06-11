import torch
from torch.utils.data.dataloader import DataLoader

from metrics import MaskMetric, StepData


def test_mask_metric(dataset):
    metric = MaskMetric(
        name="MaskMetric", class_labels=dataset.class_map
    )

    dataloader = DataLoader(dataset, batch_size=4)
    batch = next(iter(dataloader))
    perfect_out = torch.nn.functional.one_hot(batch["target"])[
        ..., :133
    ]
    # Model output shape is [B, C, H, W]
    perfect_out = perfect_out.permute(0, 3, 1, 2)

    model_prediction = {"out": perfect_out}
    step_data = StepData(
        batch=batch, model_out=model_prediction, loss=torch.tensor(0.0)
    )

    metric.update(step_data)

    measure = metric.compute()
    assert all(
        (
            mask._masks["predictions"]._val["mask_data"]
            == mask._masks["ground_truth"]._val["mask_data"]
        ).all()
        for mask in measure["MaskMetric"]
    )
