from typing import List, Optional
import wandb
import json
import pandas as pd
from wandb.apis.public.runs import Run

# Project is specified by <entity/project-name>
PROJECT_NAME = "dikvangenuchten/MasterThesis"

WANDB_API = None


def get_runs_from_group(
    group: str, api: Optional[wandb.Api] = None
) -> List[Run]:
    if api is None:
        global WANDB_API
        if WANDB_API is None:
            WANDB_API = wandb.Api()
        api = WANDB_API
    runs = api.runs(PROJECT_NAME, filters={"group": group})
    return runs


def visualize_runs(runs):
    sweep_sum = []
    for run in runs:
        config = json.loads(run.json_config)

        eval_jaccard = run.summary.get("Eval Jaccard Index", None)
        train_jaccard = run.summary.get("Train Jaccard Index", None)

        name = "var-unet"
        if all(config["model"]["value"]["skip_connections"]):
            name += "-full-skip"
        else:
            name += "-bottle"

        if all(
            config["model"]["value"]["variational_skip_connections"]
        ):
            name += "full-var"
        elif config["model"]["value"]["variational_skip_connections"][
            0
        ]:
            name += "bottle-var"
        elif not any(
            config["model"]["value"]["variational_skip_connections"]
        ):
            name += "not-var"

        if run.summary["_step"] < 14000:
            name += "*"

        sweep_sum.append(
            {
                "model_name": name,
                "eval_jaccard": eval_jaccard,
                "train_jaccard": train_jaccard,
            }
        )

    pd.DataFrame(sweep_sum)
    pass


if __name__ == "__main__":
    api = wandb.Api()

    runs = get_runs_from_group(api, "SkipImportance-24-01-06")
    visualize_runs(runs)
