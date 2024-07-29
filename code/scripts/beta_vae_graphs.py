import json
import os

import pandas as pd
import tqdm
from matplotlib import pyplot as plt

import wandb_utils

FIGURES_DIR = "../thesis/figures/beta-vae/"


def main(group: str):
    runs = wandb_utils.get_runs_from_group(group)
    runs = list(runs)

    metrics = []
    for run in tqdm.tqdm(runs):
        name = run.name
        config = json.loads(run.json_config)
        beta = config["loss"]["value"]["losses"]["kl_divergence"][
            "start_value"
        ]
        weights = config["model"]["value"]["encoder_weights"]
        kl_div = run.summary["valAnnealingWeightedLoss"] / beta
        recon = run.summary["valL1Loss"]

        metrics.append(
            {
                "name": name,
                "beta": beta,
                "weights": weights,
                "kl_div": kl_div,
                "recon": recon,
            }
        )

        # Get the last batch of images
        file_dir = os.path.join(
            FIGURES_DIR, "samples", f"{weights}-b{beta}"
        )
        download_last_eval_images(run, file_dir)

    df = pd.DataFrame(metrics)
    plot_beta_influence(df)


def download_last_eval_images(run, file_dir):
    last_step = run.summary["_step"]
    step = list(
        run.scan_history(
            keys=["EvalReconstruction"], min_step=last_step - 1
        )
    )[0]
    images = run.files(step["EvalReconstruction"]["filenames"])

    for i, file in enumerate(
        tqdm.tqdm(images, desc="Downloading recon images", leave=False)
    ):
        file.name = f"{i}.png"
        file.download(file_dir, exist_ok=True)


def plot_beta_influence(df):
    # Filter the DataFrame to get 'None' and 'Imagenet' weights
    df = df.sort_values("beta")
    color_map = {"None": "blue", "Imagenet": "orange"}
    df_none = df[df["weights"] == "None"]
    df_imgnet = df[df["weights"] == "Imagenet"]

    # Create a figure with two subplots, one for KL Div and one for Recon values
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Create a twin x-axis for the right y-axis

    for df in [df_none, df_imgnet]:
        weights = df["weights"].iloc[0]
        df.sort_values("beta")
        ax1.plot(
            df["beta"],
            df["kl_div"],
            label=f"{weights}",
            color=color_map[weights],
        )
        ax2.plot(
            df["beta"],
            df["recon"],
            label=f"{weights}",
            linestyle="--",
            color=color_map[weights],
        )

    # Set the limits of both axes to start from 0
    ax1.set_ylim([0, None])
    ax2.set_ylim([0, None])

    plt.xscale("log")
    plt.xlabel("Beta")
    ax1.set_ylabel("KL-Divergence")
    ax2.set_ylabel("L1 Loss (Reconstruction)")
    plt.title("Effect of Beta on KL Divergence and Reconstruction Loss")
    plt.legend()
    path = os.path.join(FIGURES_DIR, "beta-vae-loss.png")
    plt.savefig(path)


if __name__ == "__main__":
    group_name = "beta_vae_2024-07-27:23-33-45"
    main(group_name)
