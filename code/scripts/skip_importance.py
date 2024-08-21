"""Analysis of the Skip Importance runs
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

FIGURES_DIR = "../thesis/figures/skip_importance/"
APPENDIX_DIR = "../thesis/appendices/"


def main(group):
    runs = wandb_utils.get_runs_from_group(group)
    runs = list(runs)
    metrics = get_metrics(runs)
    print(metrics)
    create_metrics_table(metrics.copy())
    analyze_metrics(metrics.copy())


def create_metrics_table(metrics: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    metrics["Evaluation Jaccard Index"] = metrics["eval_metric"]
    table = metrics.pivot_table(
        index="skip_type",
        columns="skip_num",
        values=[
            "Evaluation Jaccard Index",
            "Parameters (x$1e^6$)",
            "Total MAC (x$1e^9$)",
        ],
    )
    table.index = table.index.rename("skip\_type")
    table.columns = table.columns.rename([None, "skip\_num"])
    table = table.transpose()
    caption = "Evaluation Jaccard Index for the VAES using different amounts and types of skip connections."
    table.style.format(precision=3, escape="latex").highlight_max(
        props="textbf:--rwrap;",
        subset=("Evaluation Jaccard Index",),
        axis=1,
    ).highlight_min(
        props="textbf:--rwrap;",
        subset=("Parameters (x$1e^6$)",),
        axis=1,
    ).highlight_min(
        props="textbf:--rwrap;", subset=("Total MAC (x$1e^9$)",), axis=1
    ).to_latex(
        os.path.join(FIGURES_DIR, "skip-importance-results.tex"),
        caption=caption,
        label="tab:skip_results",
        position="ht",
        clines="skip-last;data",
        position_float="centering",
        multirow_align="t",
        hrules=True,
    )


def analyze_metrics(metrics: pd.DataFrame):
    metric = "eval_metric"
    factors = "skip_type * skip_num"
    all_parameter_model = ols(
        " ~ ".join([metric, factors]), data=metrics
    ).fit()
    anova = anova_lm(all_parameter_model)

    # Format anova table and save it as tex
    caption = """ANOVA results estimating the influence of each parameter.\\\\Where: \\\\\\hphantom{tabb}`skip_type' is the type of skip connection used.\\\\\\hphantom{tabb}`skip_num' the number of skip connections.\\\\\\hphantom{tabb}$A$:$B$ is the interaction effect between $A$ and $B$"""
    caption = caption.replace("_", "\_")
    label = "tab:skip_importance_anova_all"
    anova_tex = utils.format_anova_table(
        anova, caption=caption, label=label
    )

    significant_effects = anova.index[anova["PR(>F)"] < 0.05]
    rhs = " + ".join(significant_effects)
    most_likely_model = " ~ ".join([metric, rhs])
    most_likely_summary = (
        ols(formula=most_likely_model, data=metrics).fit().summary2()
    )

    # Format effect table
    caption = "Coefficients of the OLS showing the influence of the hyperparameters on the Evaluation Jaccard Index."
    label = "tab:skip_importance_ols_effects"
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
        os.path.join(APPENDIX_DIR, "skip_importance.tex"), mode="w"
    ) as f:
        f.write(
            "\\chapter{Skip Importance OLS}\n\\label{appendix:skip_importance_full}"
        )
        f.write("\n")
        f.write(
            " ".join(
                [
                    "This is the summary made of the OLS model by the Python Package: Statsmodels~\\cite{josef_perktold_2024_10984387}.",
                    "First an OLS model containing all 1 and 2 level interaction effects was fitted.",
                    "This was then analysed using `anova\_lm'. All significant ($\\alpha\\le0.05$) effects where used in the final model.",
                    "The full summary of which can be seen in Table~\\ref{tab:skip_importance_full_ols}.\n\n",
                ]
            )
        )
        table_text = most_likely_summary.as_latex(
            label="tab:skip_importance_full_ols"
        )
        table_text = table_text.replace(
            "begin{table}", "begin{table}[ht]"
        )
        f.write(table_text)


def get_metrics(runs: List[wandb_utils.Run]) -> pd.DataFrame:
    metrics = []
    for run in tqdm.tqdm(runs):
        config = json.loads(run.json_config)
        architecture = get_architecture(config)
        eval_metric = get_eval_metric(run)
        skip_connections = config["model"]["value"]["skip_connections"]
        skip_type = skip_connections[0]
        skip_num = sum(1 for t in skip_connections if t == skip_type)

        config["model"]["value"]["label_channels"] = 25
        # model = utils.instantiate_dict(
        # config["model"]["value"]
        # )
        # del model
        summary = utils.get_model_summary_in_subprocess(
            config["model"]["value"]
        )

        metrics.append(
            {
                "architecture": architecture,
                "skip_type": skip_type,
                "skip_num": skip_num,
                "eval_metric": eval_metric,
                "Parameters (x$1e^6$)": summary["total_params"] * 1e-6,
                "Total MAC (x$1e^9$)": summary["total_mult_adds"]
                * 1e-9,
            }
        )
    return pd.DataFrame(metrics)


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
    main("skip_importance_2024-08-02:09-18-25")
