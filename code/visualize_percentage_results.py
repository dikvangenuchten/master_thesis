import pandas as pd
import wandb
import json
import tqdm

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("dikvangenuchten/MasterThesis")

summary_list, config_list, name_list = [], [], []
for run in tqdm.tqdm(runs):
    # Filter on the specific multirun
    if run.metadata is None or (
        run.metadata.get("args", [""])[0]
        != "/home/mcs001/20182591/master_thesis/code/src/multirun/2024-06-17/09-55-18/.submitit/%j"
    ):
        continue
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    config = json.loads(run.json_config)

    percentage = config["dataset"]["value"]["percentage"]
    eval_jaccard = run.summary["Eval Jaccard Index"]
    train_jaccard = run.summary["Train Jaccard Index"]
    if config["model"]["value"]["_target_"] == "models.MobileVAE":
        if config["model"]["value"]["load_encoder"] is True:
            name = "VAE_pretrained"
        else:
            name = "VAE"
    elif config["model"]["value"]["_target_"] == "models.UNet":
        name = "UNet"
    elif (
        config["model"]["value"]["_target_"] == "models.VariationalUNet"
    ):
        name = "Var UNet"
    else:
        raise RuntimeError(
            f"Unknown model target: {config['model']['value']['_target_']}"
        )
    if config["model"]["value"]["encoder_weights"] == "imagenet":
        name += "-imagenet"

    summary_list.append(
        {
            "model_name": name,
            "eval_jaccard": eval_jaccard,
            "train_jaccard": train_jaccard,
            "dataset percentage": percentage,
        }
    )


ax = (
    pd.DataFrame(summary_list)
    .pivot(
        index="dataset percentage",
        columns="model_name",
        values="eval_jaccard",
    )
    .plot(legend=True, logx=True)
)
# We want descending order of dataset percentage
ax.invert_xaxis()
ax.legend(loc="best")
ax.set_title("Evaluation Percentage")
ax.get_figure().savefig("eval_percentages.png")


ax = (
    pd.DataFrame(summary_list)
    .pivot(
        index="dataset percentage",
        columns="model_name",
        values="train_jaccard",
    )
    .plot(legend=True, logx=True)
)
# We want descending order of dataset percentage
ax.invert_xaxis()
ax.legend(loc="best")
ax.set_title("Training Jaccard Index")
ax.get_figure().savefig("train_percentages.png")
