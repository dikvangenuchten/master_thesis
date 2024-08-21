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
STEP = 9800


def main_recon(project, group):
    runs = wandb_utils.get_runs_from_group(group, project_name=project)
    runs = list(runs)
    finished_runs = [r for r in runs if r.state != "running"]
    if len(finished_runs) != len(runs):
        print("Warning: Not all runs are finished yet")
        runs = finished_runs
    dir = os.path.join(FIGURES_DIR, "reconstruction")
    os.makedirs(dir, exist_ok=True)

    metrics = get_metrics_recon(runs)
    # Do analysis and make plots for metrics
    create_tables_recon(metrics, dir)
    # analyze_metrics(metrics)

    download_samples(runs, dir, "EvalReconstruction")


def main_semantic(project, group):
    runs = wandb_utils.get_runs_from_group(group, project_name=project)
    runs = list(runs)
    finished_runs = [r for r in runs if r.state != "running"]
    if len(finished_runs) != len(runs):
        print("Warning: Not all runs are finished yet")
        runs = finished_runs
    dir = os.path.join(FIGURES_DIR, "semantic")
    os.makedirs(dir, exist_ok=True)

    metrics = get_metrics_semantic(runs)
    # Do analysis and make plots for metrics
    create_tables_semantic(metrics, dir)
    download_samples(runs, dir, "EvalMask")


def get_metrics_recon(runs: List[wandb_utils.Run]) -> pd.DataFrame:
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
                "L1Loss": eval_metric * (128 * 128),
                "KL-Divergence Per Pixel": kl_div / (128 * 128),
            }
        )
    return pd.DataFrame(metrics)


def get_metrics_semantic(runs: List[wandb_utils.Run]) -> pd.DataFrame:
    metrics = []
    for run in runs:
        config = json.loads(run.json_config)
        backbone = get_backbone(config)
        eval_metric = get_eval_metric(run)
        model_stats = get_model_summary(config["model"]["value"])

        metrics.append(
            {
                "Backbone": backbone,
                "Parameters ($1e^6$)": model_stats.total_params / 1e6,
                "MAC ($1e^9$)": model_stats.total_mult_adds / 1e9,
                "Jaccard Index": eval_metric,
            }
        )
    return pd.DataFrame(metrics)


def create_tables_semantic(metrics: pd.DataFrame, dir: str):
    # Create results table
    caption = "VAES results of the various backbones."
    label = "tab:vaes-backbones-results"
    metrics.set_index("Backbone", inplace=True)
    metrics.sort_index(inplace=True)
    tex = (
        metrics.style.format(precision=2)
        .highlight_min(
            subset=["Parameters ($1e^6$)", "MAC ($1e^9$)"],
            props="textbf:--rwrap;",
        )
        .highlight_max(
            subset=["Jaccard Index"],
            props="textbf:--rwrap;",
        )
        .to_latex(
            caption=caption,
            label=label,
            position="ht",
            position_float="centering",
            hrules=True,
        )
        .replace("mobilevitv2_100", "MobileViT")
        .replace("mobilenetv2_100", "MobileNetV2")
        .replace("efficientnet_b2", "EfficientNet")
        .replace("resnet", "ResNet")
        .replace("_", "\_")
    )
    with open(os.path.join(dir, "backbones_vae.tex"), mode="w") as file:
        file.write(tex)


def create_tables_recon(metrics: pd.DataFrame, dir: str):
    # Create results table
    caption = "VAE results of the various backbones."
    label = "tab:vae-backbones-results"
    metrics.set_index("Backbone", inplace=True)
    metrics.sort_index(inplace=True)
    tex = (
        metrics.style.format(precision=2)
        .highlight_min(
            props="textbf:--rwrap;",
        )
        .to_latex(
            caption=caption,
            label=label,
            position="ht",
            position_float="centering",
            hrules=True,
        )
        .replace("mobilevitv2_100", "MobileViT")
        .replace("mobilenetv2_100", "MobileNetV2")
        .replace("efficientnet_b2", "EfficientNet")
        .replace("resnet", "ResNet")
        .replace("_", "\_")
    )
    with open(os.path.join(dir, "backbones_vae.tex"), mode="w") as file:
        file.write(tex)


def get_loss_components(run):
    split = "val"
    components = [
        f"{split}L1Loss",
        f"{split}AnnealingWeightedLoss",
    ]
    result = next(run.scan_history(keys=components, min_step=STEP))
    l1loss = result[components[0]]
    klloss = result[components[1]]
    return l1loss, klloss


def download_samples(runs: List[wandb_utils.Run], dir: str, key: str):
    for run in tqdm.tqdm(runs, desc="Downloading samples per run"):
        config = json.loads(run.json_config)
        backbone = get_backbone(config)

        file_dir = os.path.join(dir, "samples", f"{backbone}")
        wandb_utils.download_last_eval_images(
            run, file_dir, key=key, step=STEP
        )


def get_backbone(config: dict):
    return config["model"]["value"]["encoder_name"]


def get_model_summary(model_config, input_shape=(3, 128, 128)):
    model = utils.instantiate_dict(model_config, label_channels=3)
    return utils.get_model_summary(model)


def get_eval_metric(run):
    if isinstance(run.summary["EvalMetric"], float):
        return run.summary["EvalMetric"]
    return run.summary["EvalMetric"]["L1-Loss"]


if __name__ == "__main__":
    # project_name = "dikvangenuchten/MasterThesis-Public"
    project_name = None
    group_name = "backbone_sweep_recon_2024-08-05:13-32-29"
    main_recon(project_name, group_name)
    group_name = "backbone_sweep_semantic_2024-08-05:13-33-04"
    main_semantic(project_name, group_name)
