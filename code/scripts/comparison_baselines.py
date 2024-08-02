"""Script to generate plots and tables for the comparison with the baseline.
"""

import json
import os
from typing import List
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import tqdm
import wandb_utils
import utils

FIGURES_DIR = "../thesis/figures/baselines/"
APPENDIX_DIR = "../thesis/appendices/"


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
        caption="The Evaluation Jaccard Index for our model and the baselines for various parameters. The higher the score, the better.",
        label="tab:baseline_results",
        position="ht",
        hrules=True,
    )


def analyze_metrics(metrics: pd.DataFrame):
    metrics = metrics.copy()
    lhs = "eval_metric"
    # Because we did not run frozen=True + weights=None we cannot
    # do a full analysis, hence we skip 3-way interaction
    rhs = "(frozen + weights + architecture) ** 2"
    formula = " ~ ".join([lhs, rhs])
    all_parameter_model = ols(formula=formula, data=metrics).fit()
    anova = anova_lm(all_parameter_model)

    # Format anova table and save it as tex
    caption = """Anova results estimating the influence of each parameter.\\\\Where: \\\\\\hphantom{tabb}`weights' are the pretrained weights (or lack thereof) used.\\\\\\hphantom{tabb}`architecture' is the model architecture used.\\\\\\hphantom{tabb}'frozen' indicates whether the encoder was frozen\\\\\\hphantom{tabb}$A$:$B$ is the interaction effect between $A$ and $B$"""
    label = "tab:comparison_baselines_anova_all"
    anova_tex = utils.format_anova_table(
        anova, caption=caption, label=label
    )

    # Select the effects that have a P-value < 0.05
    significant_effects = anova.index[anova["PR(>F)"] < 0.05]
    rhs = " + ".join(significant_effects)
    most_likely_model = " ~ ".join([lhs, rhs])
    most_likely_summary = (
        ols(formula=most_likely_model, data=metrics, cov_type="HC3")
        .fit()
        .summary2()
    )

    # Format effect table
    caption = "Coefficients of the OLS showing the influence of the hyperparameters on the Evaluation Jaccard Index."
    label = "tab:comparison_baselines_ols_effects"
    effect_tex = utils.format_effect_size(
        most_likely_summary.tables[1], caption=caption, label=label
    )

    # Add the anova and effect table to figures
    with open(os.path.join(FIGURES_DIR, "tables.tex"), mode="w") as f:
        f.write(anova_tex)
        f.write("\n\n")
        f.write(effect_tex)

    # Create an appendix containing the full results
    with open(
        os.path.join(APPENDIX_DIR, "comparison_baselinse.tex"), mode="w"
    ) as f:
        f.write(
            "\\chapter{Baseline Comparison OLS}\n\\label{appendix:baselines_comparison_full}"
        )
        f.write("\n")
        f.write(
            " ".join(
                [
                    "This is the of summary made the OLS model by the Python Package: Statsmodels~\\cite{josef_perktold_2024_10984387}.",
                    "First an OLS model containing all 1 and 2 level interaction effects was fitted.",
                    "This was then analysed using `anova\_lm'. All significant ($\\alpha\\le0.05$) effects where used in the final model.",
                    "The full summary of which can be seen in Table~\\ref{tab:comparison_baselines_full_ols}.\n\n"
                ]
            )
        )
        table_text = most_likely_summary.as_latex(label="tab:comparison_baselines_full_ols")
        table_text = table_text.replace("begin{table}", "begin{table}[ht]")
        f.write(table_text)


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
