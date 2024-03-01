import json
import os
from typing import Callable, Optional, Literal, Tuple

import torch
from PIL import Image as PImage
from torchvision.tv_tensors import Image, Mask
from pycocotools.coco import COCO


class CoCoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal["train", "val"] = "val",
        root: Optional[str] = None,
        annFile: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        base_path = "/datasets/coco/"

        self._image_root = os.path.join(base_path, f"{split}2017/")
        self._panoptic_root = os.path.join(
            base_path, "annotations", f"panoptic_{split}2017"
        )

        with open(self._panoptic_root + ".json") as f:
            self._panoptic_anns = json.load(f)

        self._coco_path = os.path.join(
            base_path, "annotations", f"instances_{split}2017.json"
        )
        self._coco = COCO(self._coco_path)
        self._ids = list(sorted(self._coco.imgs.keys()))

        self.transform = transform

        self._cat_id_to_semantic = {
            cat["id"]: i for i, cat in enumerate(self._panoptic_anns["categories"])
        }
        self._class_map = {
            i: cat["name"] for i, cat in enumerate(self._panoptic_anns["categories"])
        }

        self.ignore_index = len(self.class_map)

    def __len__(self) -> int:
        return len(self._panoptic_anns["annotations"])

    @property
    def class_map(self) -> dict:
        return self._class_map

    def _load_image(self, id: int) -> Image:
        img_id = self._panoptic_anns["annotations"][id]["image_id"]
        path = self._coco.imgs[img_id]["file_name"]
        return Image(PImage.open(os.path.join(self._image_root, path)).convert("RGB"))

    def _load_panoptic_mask(self, id: int) -> Mask:
        """Load the panoptic mask for `id`

        Based on the data format specified by [COCO](https://cocodataset.org/#format-data)

        Args:
            id (int): The 'id' (index) of the mask

        Returns:
            Mask: A panoptic mask in the shape of [2, W, H]
        """
        ann = self._panoptic_anns["annotations"][id]
        path = ann["file_name"]
        mask = Image(
            PImage.open(os.path.join(self._panoptic_root, path)).convert("RGB")
        )
        # Unlabeled places are given the 0 class

        instance_mask = rgb2id(mask)
        sem_mask = torch.zeros_like(instance_mask, dtype=torch.long)
        for segment_info in ann["segments_info"]:
            sem_mask[instance_mask == segment_info["id"]] = self._cat_id_to_semantic[
                segment_info["category_id"]
            ]
        # Temporarily only load semantic mask

        # Set unlabeled values
        sem_mask[mask.sum(0) == 0] = len(self.class_map)
        return Mask(sem_mask.unsqueeze(0))

        ins_mask = torch.zeros_like(instance_mask, dtype=torch.long)
        for new_id, old_id in enumerate(instance_mask.unique()):
            ins_mask[instance_mask == old_id] = new_id
        return Mask(torch.stack([sem_mask, ins_mask]))

    def __getitem__(self, index) -> Tuple[Image, Mask]:
        img = self._load_image(index)
        panoptic_mask = self._load_panoptic_mask(index)

        if self.transform is not None:
            img, semantic_mask = self.transform(img, panoptic_mask)
        return img, semantic_mask


def rgb2id(color: torch.Tensor):
    """Convert an RGB value to an id

    Using the formula: ids=R+G*256+B*256^2

    Args:
        color (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    color = color.to(dtype=torch.long)
    return color[0, :, :] + 256 * color[1, :, :] + 256 * 256 * color[2, :, :]


if __name__ == "__main__":
    ds = CoCoDataset()
    print(ds[0])
