import json
import os
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Literal,
    get_args,
)
from functools import cache

import torch
from diffusers.models.autoencoders.vae import (
    DiagonalGaussianDistribution,
)
from PIL import Image as PImage
from torchvision.tv_tensors import Image, Mask
from pycocotools.coco import COCO
import tqdm

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
        dataset_root: str = "/datasets/",
        rel_path: str = "coco/",
        transform: Optional[Callable] = None,
        top_k_classes: Optional[int] = None,
        supercategories_only: bool = False,
        sample: bool = True,
        length: Optional[int] = None,
        ignore_index=None,
    ):
        self._length = length

        # The absolute path to the dataset
        base_path = os.path.join(dataset_root, rel_path)

        unsuported_outs = {
            k: v
            for k, v in output_structure.items()
            if v not in get_args(OUTPUT_TYPES)
        }

        assert (
            len(unsuported_outs) == 0
        ), f"The following outputs are not supported: {unsuported_outs} not in {get_args(OUTPUT_TYPES)}"

        self.output_structure = output_structure

        self._image_root = os.path.join(base_path, f"{split}2017/")
        self._panoptic_root = os.path.join(
            base_path, "annotations", f"panoptic_{split}2017"
        )
        self._latent_root = os.path.join(base_path, f"{split}_latents")

        with open(self._panoptic_root + ".json") as f:
            self._panoptic_anns = json.load(f)

        self._panoptic_anns = _filter_annotations(
            self._panoptic_anns,
            top_k_classes=top_k_classes,
            supercategories_only=supercategories_only,
        )

        self._coco_path = os.path.join(
            base_path, "annotations", f"instances_{split}2017.json"
        )
        self._coco = COCO(self._coco_path)
        self._ids = list(sorted(self._coco.imgs.keys()))

        self.transform = transform

        self._cat_id_to_semantic = {
            cat["id"]: i
            for i, cat in enumerate(self._panoptic_anns["categories"])
        }
        self._class_map = {
            i: cat["name"]
            for i, cat in enumerate(self._panoptic_anns["categories"])
        }

        if ignore_index is None:
            ignore_index = len(self.class_map)
        elif ignore_index < len(self.class_map):
            print("Warning: ignore index is < the amount of classes")

        self.ignore_index = ignore_index
        self._sample = sample
        self._weights = None

    def __len__(self) -> int:
        return len(self._panoptic_anns["annotations"])

    @property
    def class_map(self) -> dict:
        if self.output_structure.get("target") == "img":
            return {0: "R", 1: "G", 3: "B"}
        return self._class_map

    @property
    def class_weights(self) -> List[float]:
        if self._weights is None:
            frequencies = [0] * (
                max(self._cat_id_to_semantic.values()) + 1
            )
            for annotation in tqdm.tqdm(
                self._panoptic_anns["annotations"][: len(self)]
            ):
                for segment in annotation["segments_info"]:
                    id = self._cat_id_to_semantic[
                        segment["category_id"]
                    ]
                    frequencies[id] += segment["area"]
            self._weights = [
                sum(frequencies) / (freq * len(frequencies))
                if freq != 0
                else 1
                for freq in frequencies
            ]
        return self._weights

    @cache
    def _load_latent(self, idx: int) -> torch.Tensor:
        parameters: torch.Tensor = torch.load(
            os.path.join(self._latent_root, f"vae_latent_{idx}.pt")
        )
        if self._sample:
            dist = DiagonalGaussianDistribution(
                parameters=parameters.unsqueeze(0)
            )
            return LatentTensor(dist.rsample().squeeze(0))
        else:
            mean, _logvar = parameters.chunk(parameters, 2, dim=0)
            return LatentTensor(mean)

    def _load_image(self, idx: int) -> Image:
        img_id = self._panoptic_anns["annotations"][idx]["image_id"]
        path = self._coco.imgs[img_id]["file_name"]
        return Image(
            PImage.open(os.path.join(self._image_root, path)).convert(
                "RGB"
            )
        )

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
            PImage.open(
                os.path.join(self._panoptic_root, path)
            ).convert("RGB")
        )
        # Unlabeled places are given the 0 class

        instance_mask = rgb2id(mask)
        sem_mask = (
            torch.ones_like(instance_mask, dtype=torch.long)
            * self.ignore_index
        )
        for segment_info in ann["segments_info"]:
            sem_mask[
                instance_mask == segment_info["id"]
            ] = self._cat_id_to_semantic[segment_info["category_id"]]

        # 'Temporarily' only load semantic mask
        return Mask(sem_mask)

        ins_mask = torch.zeros_like(instance_mask, dtype=torch.long)
        for new_id, old_id in enumerate(instance_mask.unique()):
            ins_mask[instance_mask == old_id] = new_id
        return Mask(torch.stack([sem_mask, ins_mask]))

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        out = self._getitem(index)
        if self.transform is not None:
            return self.transform(out)
        return out

    def _getitem(self, index) -> Dict[str, torch.Tensor]:
        if self._length is not None:
            # This is for some reason ~8 times faster compared to reducing the actual dataset size
            index = index % self._length
        return {
            k: self._get_type(index, v)
            for k, v in self.output_structure.items()
        }

    def _get_type(self, index, type_) -> torch.Tensor:
        if type_ == "img":
            return self._load_image(index)
        elif type_ == "latent":
            return self._load_latent(index)
        elif type_ == "semantic_mask":
            return self._load_panoptic_mask(index)
        else:
            raise RuntimeError(
                f"{type_} is not supported in {self.__qualname__}"
            )


def _filter_annotations(
    data: Dict, top_k_classes: Optional[int], supercategories_only: bool
) -> Dict:
    if supercategories_only:
        id_to_supercategory = {
            c["id"]: c["supercategory"] for c in data["categories"]
        }
        supercategories = list(
            {c["supercategory"] for c in data["categories"]}
        )

        for annotation in tqdm.tqdm(data["annotations"]):
            for segment in annotation["segments_info"]:
                segment["category_id"] = supercategories.index(
                    id_to_supercategory[segment["category_id"]]
                )

        super_isthing = {
            c["supercategory"]: c["isthing"] for c in data["categories"]
        }
        data["categories"] = [
            {
                "supercategory": category,
                "isthing": super_isthing[category],
                "name": category,
                "id": i,
            }
            for i, category in enumerate(supercategories)
        ]

    if top_k_classes is None:
        return data
    # Calculate the top k classes
    freq_dict = {}
    for annotation in tqdm.tqdm(data["annotations"]):
        for segment in annotation["segments_info"]:
            id = segment["category_id"]
            freq_dict[id] = freq_dict.get(id, 0) + segment["area"]

    top_k = sorted(
        ((v, k) for k, v in freq_dict.items()), reverse=True
    )[:top_k_classes]
    top_k_cat_ids = [id for (freq_, id) in top_k]

    data["categories"] = [
        d for d in data["categories"] if d["id"] in top_k_cat_ids
    ]
    # Add a 'Background' class
    data["categories"].append(
        {
            "supercategory": "background",
            "isthing": 0,
            "id": top_k_classes,
            "name": "background",
        }
    )
    annotations = []
    for annotation in tqdm.tqdm(data["annotations"]):
        filtered_annotation = annotation.copy()
        filtered_annotation["segments_info"] = []
        for segment in annotation["segments_info"]:
            if segment["category_id"] in top_k_cat_ids:
                filtered_annotation["segments_info"].append(segment)
            else:
                segment["category_id"] = top_k_classes
                filtered_annotation["segments_info"].append(segment)
        if len(filtered_annotation["segments_info"]) > 0:
            annotations.append(filtered_annotation)
    data["annotations"] = annotations
    return data


def rgb2id(color: torch.Tensor):
    """Convert an RGB value to an id

    Using the formula: ids=R+G*256+B*256^2

    Args:
        color (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    color = color.to(dtype=torch.long)
    return (
        color[0, :, :]
        + 256 * color[1, :, :]
        + 256 * 256 * color[2, :, :]
    )


if __name__ == "__main__":
    ds = CoCoDataset()
    print(ds[0])
