import json
import os
import logging

from tqdm import tqdm
from typing import Literal, Optional, List
from PIL import Image
import numpy as np

try:
    import fiftyone

    fiftyone.config.do_not_track = True
except ImportError as e:
    logging.error(
        "Could not import fiftyone, please install using `pip install fiftyone`"
    )
    raise e from None


def convert_data(sample):
    filepath = sample.filepath
    dir_, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)

    img = Image.open(filepath)
    labels = ["Void"]
    mask = np.zeros(img.size[::-1], dtype=np.uint8)
    for i, detection in enumerate(sample.ground_truth.detections, 1):
        labels.append(detection.label)
        mask += detection.to_segmentation(
            frame_size=img.size, target=i
        ).mask

    maskpath = os.path.join(dir_, f"{filename}-mask.png")
    mask_img = Image.fromarray(mask.swapaxes(1, 0))
    mask_img.save(maskpath)

    # Convert mask to png
    return {
        "filepath": filepath,
        "maskpath": maskpath,
        "labels": labels,
    }


def main(
    name: str,
    split: Literal["train", "validation"] = "validation",
    max_samples: Optional[int] = None,
    classes: List[str] = ["Cat", "Dog"],
):
    fiftyone.config.dataset_zoo_dir = "/datasets/fiftyone/"
    fiftyone.config.default_dataset_dir = "/datasets/fiftyone/"

    dataset = fiftyone.zoo.load_zoo_dataset(
        "open-images-v7",
        split=split,
        classes=classes,
        label_types=["segmentations"],
        max_samples=max_samples,
        persistent=True,
    )
    data = [convert_data(x) for x in tqdm(dataset)]
    dataset_dir = "/datasets/custom/"
    os.makedirs(dataset_dir, exist_ok=True)
    with open(
        os.path.join(dataset_dir, f"{name}_{split}.json"), mode="w"
    ) as f:
        json.dump(data, f)


if __name__ == "__main__":
    main("cat_dog_toy", split="validation")
    main("cat_dog_toy", split="train")
