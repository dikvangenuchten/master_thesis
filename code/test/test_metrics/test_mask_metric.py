import pytest
import torch
from torch.utils.data.dataloader import DataLoader

from metrics import MaskMetric, StepData


@pytest.fixture()
def class_labels():
    return {0: "person", 1: "bicycle", 2: "car"}


def test_mask_metric(dataset):
    metric = MaskMetric(
        name="MaskMetric", class_labels=dataset.class_map
    )

    dataloader = DataLoader(dataset, batch_size=4)
    batch = next(iter(dataloader))
    perfect_out = torch.nn.functional.one_hot(batch["target"])[
        ..., :133
    ]

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
