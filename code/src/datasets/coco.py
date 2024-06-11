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
import warnings

import torch
from diffusers.models.autoencoders.vae import (
    DiagonalGaussianDistribution,
)
from PIL import Image as PImage
from torchvision.tv_tensors import Image, Mask
import tqdm

from datasets import LatentTensor

OUTPUT_TYPES = Literal["img", "latent", "semantic_mask"]


def _open_cache_wrapper(fn, cache_dir):
    def _inner(path):
        cache_path = os.path.join(cache_dir, path)
        if not os.path.isfile(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(path, mode="rb") as src:
                with open(cache_path, mode="wb") as tgt:
                    tgt.write(src.read())

        return fn(cache_path)

    return _inner


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
        percentage: float = 1.00,
        cache_dir: Optional[str] = None,
    ):
        # The absolute path to the dataset
        self.base_path = os.path.join(dataset_root, rel_path)
        self._image_root = os.path.join(self.base_path, f"{split}2017/")
        self._panoptic_root = os.path.join(
            self.base_path, "annotations", f"panoptic_{split}2017"
        )
        self._latent_root = os.path.join(
            self.base_path, f"{split}_latents"
        )

        self._length = length

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self._open_pil_image = _open_cache_wrapper(
                self._open_pil_image, cache_dir
            )

        # Ignore the percentage in validation dataset
        if percentage < 1:
            if split == "val":
                warnings.warn("Ignoring percentage for validation split")
                percentage = 1
            else:
                print(f"Training will happen with a split of {percentage}")

        self.output_structure = self.parse_output_structure(
            output_structure
        )

        with open(self._panoptic_root + ".json") as f:
            panoptic_anns = json.load(f)

        self._panoptic_anns = _filter_annotations(
            panoptic_anns,
            top_k_classes=top_k_classes,
            supercategories_only=supercategories_only,
            percentage=percentage,
        )

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

        self.transform = transform
        self.ignore_index = ignore_index
        self._sample = sample
        self._weights = None

    def parse_output_structure(
        self, output_structure
    ) -> Dict[str, str]:
        unsuported_outs = {
            k: v
            for k, v in output_structure.items()
            if v not in get_args(OUTPUT_TYPES)
        }
        assert (
            len(unsuported_outs) == 0
        ), f"The following outputs are not supported: {unsuported_outs} not in {get_args(OUTPUT_TYPES)}"
        return output_structure

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
        path = self._panoptic_anns["annotations"][idx][
            "file_name"
        ].replace("png", "jpg")
        return self._open_pil_image(
            os.path.join(self._image_root, path)
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
        # Load the encoded mask
        enc_mask = self._open_pil_image(
            os.path.join(self._panoptic_root, path)
        )
        # Decode the mask
        instance_mask = rgb2id(enc_mask)
        # Unlabeled places are given the 'ignore_index' class
        sem_mask = torch.full_like(
            instance_mask, self.ignore_index, dtype=torch.long
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

    def _open_pil_image(self, path) -> Image:
        return Image(PImage.open(path).convert("RGB"))

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
    data: Dict,
    top_k_classes: Optional[int],
    supercategories_only: bool,
    percentage: float,
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

    if top_k_classes is not None:
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

    if percentage < 1:
        cur_size = len(data["annotations"])
        target_size = round(cur_size * percentage)
        data["annotations"] = data["annotations"][:target_size]

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
