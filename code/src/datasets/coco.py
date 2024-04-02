import json
import os
from typing import Callable, Dict, Optional, Literal, get_args
from functools import cache

import torch
from diffusers.models.autoencoders.vae import (
    DiagonalGaussianDistribution,
)
from PIL import Image as PImage
from torchvision.tv_tensors import Image, Mask
from pycocotools.coco import COCO

from datasets import LatentTensor

OUTPUT_TYPES = Literal["img", "latent", "semantic_mask"]


class CoCoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal["train", "val"] = "val",
        output_structure: Dict[str, OUTPUT_TYPES] = {
            "input": "latent",
            "target": "semantic_mask",
        },
        base_path: str = "/datasets/coco/",
        root: Optional[str] = None,
        annFile: Optional[str] = None,
        transform: Optional[Callable] = None,
        latents: bool = False,
        sample: bool = True,
        num_classes: Optional[int] = None,
    ):
        unsuported_outs = {
            k: v for k, v in output_structure.items() if v not in get_args(OUTPUT_TYPES)
        }

        assert (
            len(unsuported_outs) == 0
        ), f"The following outputs are not supported: {unsuported_outs}"

        self.output_structure = output_structure

        self._image_root = os.path.join(base_path, f"{split}2017/")
        self._panoptic_root = os.path.join(
            base_path, "annotations", f"panoptic_{split}2017"
        )
        self._latent_root = os.path.join(base_path, f"{split}_latents")

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
        self._latents = latents
        self._sample = sample

    def __len__(self) -> int:
        return len(self._panoptic_anns["annotations"])

    @property
    def class_map(self) -> dict:
        return self._class_map

    @cache
    def _load_latent(self, idx: int) -> torch.Tensor:
        parameters: torch.Tensor = torch.load(
            os.path.join(self._latent_root, f"vae_latent_{idx}.pt")
        )
        if self._sample:
            dist = DiagonalGaussianDistribution(parameters=parameters.unsqueeze(0))
            return LatentTensor(dist.sample().squeeze(0))
        else:
            mean, _logvar = parameters.chunk(parameters, 2, dim=0)
            return LatentTensor(mean)

    def _load_image(self, idx: int) -> Image:
        img_id = self._panoptic_anns["annotations"][idx]["image_id"]
        path = self._coco.imgs[img_id]["file_name"]
        return Image(PImage.open(os.path.join(self._image_root, path)).convert("RGB"))

    def _load_panoptic_mask(self, idx: int) -> Mask:
        """Load the panoptic mask for `id`

        Based on the data format specified by [COCO](https://cocodataset.org/#format-data)

        Args:
            idx (int): The 'idx' (index) of the mask

        Returns:
            Mask: A panoptic mask in the shape of [2, W, H]
        """
        ann = self._panoptic_anns["annotations"][idx]
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

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        out = {k: self._get_type(index, v) for k, v in self.output_structure.items()}
        if self.transform is not None:
            return self.transform(out)
        return out

    def _get_type(self, index, type_) -> torch.Tensor:
        if type_ == "img":
            return self._load_image(index)
        elif type_ == "latent":
            return self._load_latent(index)
        elif type_ == "semantic_mask":
            return self._load_panoptic_mask(index)
        else:
            raise RuntimeError(f"{type_} is not supported in {self.__qualname__}")


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
