"""Script to generate plots and tables for the comparison with the baseline.
"""

import json
import os
from typing import List
import pandas as pd
import numpy as np  # noqa: F401
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import tqdm
from matplotlib import pyplot as plt
import wandb_utils

FIGURES_DIR = "../thesis/figures/data_percentage/"
APPENDIX_DIR = "../thesis/appendices/"


def main(group):
    runs = wandb_utils.get_runs_from_group(group)
    runs = list(runs)

    metrics = get_metrics(runs)
    create_plots(metrics.copy())
    create_tables(metrics.copy())

    download_samples(runs)


def get_metrics(runs: List[wandb_utils.Run]) -> pd.DataFrame:
    metrics = []
    for i, run in enumerate(runs):
        config = json.loads(run.json_config)
        architecture = get_architecture(config)
        weights = config["model"]["value"]["encoder_weights"]
        percentage = config["dataset"]["value"]["percentage"]
        eval_metric = get_eval_metric(run)
        metrics.append(
            {
                "idx": i,
                "config": config,
                "architecture": architecture,
                "weights": weights,
                "fraction": percentage,
                "Jaccard_Index": eval_metric,
            }
        )
    return pd.DataFrame(metrics)


def create_tables(metrics):
    # Do analysis and make plots for metrics
    # Insert "all invalid runs"
    len(metrics)
    parameter_influence = ols(
        formula="Jaccard_Index ~ np.log10(fraction) * weights * architecture",
        data=metrics,
        cov_type="hc3",
    ).fit()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(
        os.path.join(FIGURES_DIR, "parameter_significance_table.tex"),
        mode="w",
    ) as file:
        caption = """Anova results estimating the influence of each parameter.\\\\Where: \\\\\\hphantom{tabb}'weights' are the pretrained weights (or lack thereof) used.\\\\\\hphantom{tabb}'architecture' is the model architecture used.\\\\\\hphantom{tabb}'$log_{10}(\\text{fraction})$' is the fraction of data used in log scale."""
        table = anova_lm(parameter_influence, type=2).to_latex(
            caption=caption,
            label="tab:data_fraction_parameter_significance",
            float_format="%.2f",
            position="ht",
        )
        table = table.replace(
            "np.log10(fraction)", "$log_{10}(\\text{fraction})$"
        )
        file.write(table)

    best_fit = ols(
        formula="Jaccard_Index ~ np.log10(fraction) + weights + architecture",
        data=metrics,
        cov_type="hc3",
    ).fit()
    summary = best_fit.summary2(
        title="Summary of the OLS of the most likely model.",
        float_format="%.3f",
    )
    summary.settings[0]["float_format"] = "%.3f"
    summary.tables = summary.tables[:2]
    summary.extra_txt = []
    with open(
        os.path.join(FIGURES_DIR, "parameter_influence_table.tex"),
        mode="w",
    ) as file:
        table = summary.as_latex(
            label="tab:data_fraction_parameter_influence"
        )
        file.write(table)

    with open(
        os.path.join(FIGURES_DIR, "results_dataset_fraction.tex"),
        mode="w",
    ) as file:
        table = metrics.pivot_table(
            index=["architecture", "weights"],
            columns=["fraction"],
            values="Jaccard_Index",
        )
        # Reverse colums so 1 is first
        table = table[table.columns[::-1]]
        table.to_latex(
            file,
            caption="The evaluation Jaccard Index for the various models and dataset fractions. Higher is better.",
            position="ht",
            float_format="%.3f",
            label="tab:data_fraction_results",
        )


def create_plots(metrics: pd.DataFrame):
    colors = ["#377eb8", "#ff7f00", "#4daf4a"]
    linestyles = ["solid", "dotted", "dashed"]
    architectures = ["VAES", "unet", "fpn"]
    weights = ["None", "vae-b10", "imagenet"]

    pivot = metrics.pivot_table(
        index=["architecture", "weights"],
        columns=["fraction"],
        values="Jaccard_Index",
    )
    pivot = pivot[pivot.columns[::-1]]
    fig, ax = plt.subplots()
    for i, architecture in enumerate(architectures):
        linestyle = linestyles[i]
        for j, weight in enumerate(weights):
            color = colors[j]
            x = list(pivot.loc[architecture, weight].index)
            y = list(pivot.loc[architecture, weight])
            ax.plot(x, y, color=color, linestyle=linestyle)

    # Add the legend
    for i, architecture in enumerate(architectures):
        linestyle = linestyles[i]
        ax.plot(
            [],
            [],
            color="black",
            linestyle=linestyle,
            label=architecture,
        )
    for i, weight in enumerate(weights):
        color = colors[i]
        ax.plot(
            [], [], color=color, linestyle="", marker="o", label=weight
        )
    ax.legend(loc="upper right", ncol=2)

    ax.set_ylim([0, 0.6])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Fraction of the training dataset used")
    ax.set_ylabel("Jaccard Index")
    ax.set_title("Jaccard Index for reduced dataset size")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "line-plot.png"))


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
    group_name = "dataset_percentage_2024-07-29:11-10-29"
    main(group_name)
