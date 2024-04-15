"""Copy a very small portion of the COCO dataset for use in pipeline testing"""

import json
import os
import shutil
from tqdm import tqdm


def convert(name: str, src_dir: str, tgt_dir: str):
    with open(
        os.path.join(src_dir, f"annotations/{name}_val2017.json")
    ) as f:
        instances = json.load(f)

    tiny_instances = {}
    tiny_instances["info"] = instances["info"]
    tiny_instances["licenses"] = instances["licenses"]
    tiny_instances["categories"] = instances["categories"]

    keep_image_ids = [img["id"] for img in instances["images"]][:16]

    tiny_instances["images"] = [
        img
        for img in tqdm(instances["images"], desc="Filtering images")
        if img["id"] in keep_image_ids
    ]
    tiny_instances["annotations"] = [
        ann
        for ann in tqdm(
            instances["annotations"], desc="Filtering annotations"
        )
        if ann["image_id"] in keep_image_ids
    ]

    # Copy image over

    for img in tqdm(tiny_instances["images"], desc="Copying images"):
        shutil.copy(
            os.path.join(src_dir, "val2017", img["file_name"]),
            os.path.join(tgt_dir, "val2017", img["file_name"]),
        )

    if name == "panoptic":
        os.makedirs(
            os.path.join(
                tgt_dir,
                "annotations",
                f"{name}_val2017",
            ),
            exist_ok=True,
        )
        for img in tqdm(
            tiny_instances["annotations"], desc="Copying masks"
        ):
            shutil.copy(
                os.path.join(
                    src_dir,
                    "annotations",
                    f"{name}_val2017",
                    img["file_name"],
                ),
                os.path.join(
                    tgt_dir,
                    "annotations",
                    f"{name}_val2017",
                    img["file_name"],
                ),
            )
    with open(
        os.path.join(tgt_dir, f"annotations/{name}_val2017.json"),
        mode="w",
    ) as f:
        json.dump(tiny_instances, f)


if __name__ == "__main__":
    name = "panoptic"
    # name = "instances"
    data_folder = "/datasets/coco/"
    test_data_folder = "test/data/coco/"
    for name in ["panoptic", "instances"]:
        convert(
            name=name, src_dir=data_folder, tgt_dir=test_data_folder
        )
