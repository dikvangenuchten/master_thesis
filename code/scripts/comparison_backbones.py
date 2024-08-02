"""Script to generate plots and tables for the comparison with the baseline.
"""

import json
import os
from typing import List
import pandas as pd
import tqdm
import wandb_utils
import utils

FIGURES_DIR = "../thesis/figures/vae-backbones/"
STEP = 15000


def main(project, group):
    runs = wandb_utils.get_runs_from_group(group, project_name=project)
    runs = list(runs)
    finished_runs = [r for r in runs if r.state != "running"]
    if len(finished_runs) != len(runs):
        print("Warning: Not all runs are finished yet")
        runs = finished_runs
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metrics = get_metrics(runs)
    # Do analysis and make plots for metrics
    create_tables(metrics)
    # analyze_metrics(metrics)

    download_samples(runs)


def get_metrics(runs: List[wandb_utils.Run]) -> pd.DataFrame:
    metrics = []
    for run in runs:
        config = json.loads(run.json_config)
        backbone = get_backbone(config)
        l1loss, kl_div = get_loss_components(run)
        eval_metric = get_eval_metric(run)
        model_stats = get_model_summary(config["model"]["value"])

        metrics.append(
            {
                "Backbone": backbone,
                "Parameters ($1e^6$)": model_stats.total_params / 1e6,
                "MAC ($1e^9$)": model_stats.total_mult_adds / 1e9,
                "L1Loss (Recon)": l1loss,
                "KL-Divergence": kl_div,
                "Loss": eval_metric,
            }
        )
    return pd.DataFrame(metrics)


def create_tables(metrics: pd.DataFrame):
    # Create results table
    pass


def get_loss_components(run):
    split = "val"
    components = [
        f"{split}L1Loss",
        f"{split}AnnealingWeightedLoss",
    ]
    result = next(run.scan_history(keys=components))
    l1loss = result[components[0]]
    klloss = result[components[1]]
    return l1loss, klloss


def download_samples(runs: List[wandb_utils.Run]):
    for run in tqdm.tqdm(runs, desc="Downloading samples per run"):
        config = json.loads(run.json_config)
        backbone = get_backbone(config)

        file_dir = os.path.join(FIGURES_DIR, "samples", f"{backbone}")
        wandb_utils.download_last_eval_images(
            run, file_dir, key="EvalReconstruction", step=STEP
        )


def get_backbone(config: dict):
    return config["model"]["value"]["encoder_name"]


def get_model_summary(model_config, input_shape=(3, 128, 128)):
    model = utils.instantiate_dict(
        model_config, label_channels=3
    )
    return utils.get_model_summary(model)


def get_eval_metric(run):
    eval_metric = run.summary.get("EvalMetric")
    if eval_metric is not None:
        return eval_metric
    return run.summary.get("EvalAverageLoss")


if __name__ == "__main__":
    project_name = "dikvangenuchten/MasterThesis-Public"
    group_name = "vae_backbones_2"
    main(project_name, group_name)
