"""Script to generate plots and tables for the comparison with the baseline.
"""

import json
import os
from typing import List
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import tqdm
import wandb_utils

FIGURES_DIR = "../thesis/figures/baselines/"


def main(group):
    runs = wandb_utils.get_runs_from_group(group)
    runs = list(runs)
    finished_runs = [r for r in runs if r.state != "running"]
    if len(finished_runs) != len(runs):
        print("Warning: Not all runs are finished yet")
        runs = finished_runs
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metrics = get_metrics(runs)
    # Do analysis and make plots for metrics
    create_tables(metrics)
    analyze_metrics(metrics)

    download_samples(runs)


def get_metrics(runs: List[wandb_utils.Run]) -> pd.DataFrame:
    metrics = []
    for run in runs:
        config = json.loads(run.json_config)
        architecture = get_architecture(config)
        weights = config["model"]["value"]["encoder_weights"]
        frozen = config["model"]["value"]["encoder_freeze"]
        eval_metric = get_eval_metric(run)
        metrics.append(
            {
                "architecture": architecture,
                "weights": weights,
                "frozen": frozen,
                "eval_metric": eval_metric,
            }
        )
    return pd.DataFrame(metrics)


def create_tables(metrics: pd.DataFrame):
    # Create results table
    pivot = metrics.pivot(
        index=["frozen", "architecture"],
        columns="weights",
        values="eval_metric",
    )
    pivot.style.format_index(str, level=[0]).highlight_max(
        axis=0,
        props="textbf:--rwrap;",
    ).format(
        na_rep="n.a.",
        precision=2,
    ).to_latex(
        os.path.join(FIGURES_DIR, "baselines-results.tex"),
        caption="The Evaluation Jaccard Index for our model and the baselines for various parameters. Higher is better.",
        label="tab:baseline_results",
        position="ht",
        hrules=True,
    )


def analyze_metrics(metrics: pd.DataFrame):
    print("WARNING: Removed FPN for now as it is not yet done")
    # metrics = metrics[metrics["architecture"] != "fpn"]
    print("WARNING: Removed FPN for now as it is not yet done")

    # Todo the ANOVA, we need to add the 'invalid' configs
    # for arch in metrics["architecture"].unique():
    #     metrics = metrics.append(
    #         {
    #             "architecture": arch,
    #             "weights": "None",
    #             "frozen": True,
    #             "eval_metric": 0.00,
    #         },
    #         ignore_index=True,
    #     )

    parameter_influence = ols(
        formula="eval_metric ~ weights * architecture",
        data=metrics,
        cov_type="hc1",
    ).fit()
    anova = anova_lm(parameter_influence, cov_type="hc3")
    print(anova.summary2())
    pass


def download_samples(runs: List[wandb_utils.Run]):
    for run in tqdm.tqdm(runs, desc="Downloading samples per run"):
        config = json.loads(run.json_config)
        architecture = get_architecture(config)
        weights = config["model"]["value"]["encoder_weights"]
        frozen = config["model"]["value"]["encoder_freeze"]

        file_dir = os.path.join(
            FIGURES_DIR, "samples", f"{architecture}-{weights}-{frozen}"
        )
        wandb_utils.download_last_eval_images(
            run, file_dir, key="EvalMask"
        )


def get_architecture(config: dict):
    if config["model"]["value"]["_target_"] == "models.VAES":
        return "VAES"
    else:
        return config["model"]["value"]["architecture"]


def get_eval_metric(run):
    eval_metric = run.summary.get("EvalMetric")
    if eval_metric is not None:
        return eval_metric
    return run.summary.get("Eval Jaccard Index")


if __name__ == "__main__":
    group_name = "comparison_2024-07-28:15-01-31"
    main(group_name)
