import os
from typing import List, Optional
import tqdm
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


def download_last_eval_images(run, file_dir, key="EvalReconstruction"):
    last_step = run.summary["_step"]
    step = list(run.scan_history(keys=[key], min_step=last_step - 1))[0]

    images = list(run.files(step[key]["filenames"]))
    all_masks = step[key].get("all_masks", [])
    gt_masks = [m["ground_truth"]["path"] for m in all_masks]
    pr_masks = [m["predictions"]["path"] for m in all_masks]

    for i, file in enumerate(
        tqdm.tqdm(images, desc="Downloading recon images", leave=False)
    ):
        file.name = f"{i}.png"
        file.download(file_dir, exist_ok=True)

    if len(gt_masks) > 0 and len(pr_masks) > 0:
        with open(
            os.path.join(file_dir, "class_map.json"), mode="w"
        ) as file:
            json.dump(run.config["class_map"], file)
        gt_files = list(run.files(gt_masks))
        for i, file in enumerate(
            tqdm.tqdm(
                gt_files, desc="Downloading gt_masks", leave=False
            )
        ):
            file.name = f"gt_{i}.png"
            file.download(file_dir, exist_ok=True)

        pr_files = list(run.files(pr_masks))
        for i, file in enumerate(
            tqdm.tqdm(
                pr_files, desc="Downloading pr_masks", leave=False
            )
        ):
            file.name = f"pr_{i}.png"
            file.download(file_dir, exist_ok=True)


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
