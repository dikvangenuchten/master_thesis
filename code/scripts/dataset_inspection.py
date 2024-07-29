"""Small script to generate some insights and plots of the datasets
"""
import os
import hydra
from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms import functional
import skimage

import utils  # noqa: F401

FIGURES_DIR = "../thesis/figures/datasets/"


def main(dataset_name):
    with hydra.initialize(
        config_path="../src/conf/dataset", job_name=dataset_name
    ):
        cfg = hydra.compose(
            dataset_name,
            overrides=[
                "dataset_root=/datasets/",
                "+supercategories_only=True",
            ],
        )
        train_dataset = hydra.utils.instantiate(cfg, split="train")
        val_dataset = hydra.utils.instantiate(cfg, split="val")

        train_stats = get_statistics(train_dataset)
        val_stats = get_statistics(val_dataset)

        sample_dir = os.path.join(FIGURES_DIR, dataset_name, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        generate_samples(
            train_dataset, 6, os.path.join(sample_dir, "train")
        )
        generate_samples(
            val_dataset, 6, os.path.join(sample_dir, "val")
        )

        path = os.path.join(
            FIGURES_DIR, dataset_name, "class_distribution.png"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plot_distribution(train_dataset, val_dataset, path)


def plot_distribution(train_dataset, val_dataset, file_name):
    train_freq = get_norm_freqs(train_dataset)
    val_freq = get_norm_freqs(val_dataset)
    classes = sorted(
        train_freq.keys(), key=lambda x: train_freq[x], reverse=True
    )

    fig, ax = plt.subplots(layout="constrained")
    # Plot the frequency of each class as a bar
    x = np.arange((len(classes)))
    train_freqs = [train_freq[c] * 100 for c in classes]
    val_freqs = [val_freq[c] * 100 for c in classes]
    plt.barh(
        x, train_freqs, label="Training Data", color="blue", height=0.4
    )
    plt.barh(
        [i + 0.4 for i in x],
        val_freqs,
        label="Validation Data",
        color="orange",
        height=0.4,
    )

    # Set labels for the x-axis (class names) and y-axis (frequencies)
    plt.legend(loc="upper right")
    plt.yticks(x + 0.4, classes)
    plt.ylabel("Class Names")
    plt.xlabel("Percentage")
    plt.title("Distribution of Classes in the Coco Dataset")
    plt.savefig(file_name)


def get_norm_freqs(dataset):
    distribution_of_classes = dataset.class_weights
    class_map = dataset.class_map

    class_freq = {
        v: 1 / distribution_of_classes[k] for k, v in class_map.items()
    }
    tot_freq = sum(class_freq.values())
    class_freq = {k: v / tot_freq for k, v in class_freq.items()}
    return class_freq


def modified_set3(n=27):
    base = plt.cm.Set3(np.linspace(0, 1, 12))
    return plt.cm.colors.ListedColormap(base, name="modified_set3", N=n)


def generate_samples(dataset, n, dir):
    for img_idx in range(n):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        batch = dataset[img_idx]
        img = functional.to_pil_image(batch["input"])
        ax1.imshow(img)
        mask = batch["target"]

        cmap = plt.cm.Set3(range(len(mask.unique())))
        # Set last value (i.e. unlabeled) as black
        cmap[-1] = (0, 0, 0, 1)
        # Remap mask (labels)
        new_mask = np.zeros_like(mask)
        class_map = {}
        for new, old in enumerate(mask.unique()):
            new_mask[mask == old] = new
            class_map[new] = dataset.class_map.get(
                int(old), "unlabeled"
            )

        num_classes = len(mask.unique())
        color_mask = skimage.color.label2rgb(
            new_mask, bg_label=num_classes - 1
        )
        # Get unique colors
        colors = skimage.color.label2rgb(
            np.unique(new_mask), bg_label=num_classes - 1
        )

        ax2.imshow(color_mask)

        # Insert a 'legend'
        for j in range(len(mask.unique())):
            j = int(j)
            if j != dataset.ignore_index:
                ax2.plot(
                    [],
                    [],
                    color=colors[j],
                    label=class_map[j],
                    marker="o",
                    linestyle="None",
                )
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.axis("off")
        ax2.axis("off")
        fig.tight_layout()
        os.makedirs(dir, exist_ok=True)
        fig.savefig(os.path.join(dir, f"{img_idx}.png"))
        pass
    pass


def get_statistics(dataset):
    size = len(dataset)

    samples = [dataset[i] for i in range(10)]

    pass


if __name__ == "__main__":
    main("coco")
