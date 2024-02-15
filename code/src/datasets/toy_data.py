import os
from typing import Any, Optional, Tuple
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.tv_tensors import Image, Mask


class OneColorBackground(data.Dataset):
    def __init__(
        self,
        img_size: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 0, 0),
        size: int = 100,
    ):
        super().__init__()
        self._sample = torch.ones((3, *img_size)) * torch.tensor(color).view(3, 1, 1)
        self._size = size

    def __getitem__(self, index) -> Any:
        return self._sample

    def __len__(self) -> int:
        return self._size


class SegmentationToyDataset(data.Dataset):
    """A toy dataset for image segmentation"""

    def __init__(
        self,
        background_dataset: Optional[data.Dataset] = None,
        limit: int = 100,
        split: str = "val",
    ):
        if background_dataset is None:
            background_dataset = self.default_background_dataset(split)
        self._background_dataset = background_dataset
        self._limit = limit

        self.transforms = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToDtype(torch.float32, scale=True),
                ToySegmentationTransform(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index) -> Any:
        img = self._background_dataset[index]
        if isinstance(img, tuple):
            img = img[0]

        img = Image(img)

        img, mask = self.transforms(img)
        return img, mask

    def __len__(self) -> int:
        if self._limit == -1:
            return len(self._background_dataset)
        return min(self._limit, len(self._background_dataset))

    @staticmethod
    def default_background_dataset(split: str):
        base_path = "/datasets/coco/"
        return datasets.wrap_dataset_for_transforms_v2(
            datasets.CocoDetection(
                os.path.join(base_path, f"{split}2017/"),
                os.path.join(base_path, f"annotations/instances_{split}2017.json"),
            )
        )

    def to_loader(self, *args, **kwargs) -> data.DataLoader:
        return data.DataLoader(self, *args, **kwargs)


class ToySegmentationTransform(nn.Module):
    """Augment an image with a blob ontop of it.

    Returns the modified image and the mask of the blob.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._x_size = 10
        self._y_size = 10

    def forward(self, img):
        mask = Mask(torch.zeros_like(img[0:1]))

        _, w, h = mask.shape
        x = torch.randint(self._x_size, w - self._x_size, [1])
        y = torch.randint(self._y_size, h - self._y_size, [1])

        if img.dtype == torch.uint8:
            val = 255
        else:
            val = 1.0

        img[
            0, x - self._x_size : x + self._x_size, y - self._y_size : y + self._y_size
        ] = torch.tensor([val]).reshape([1, 1, 1])
        mask[
            0, x - self._x_size : x + self._x_size, y - self._y_size : y + self._y_size
        ] = 1

        return img, mask


if __name__ == "__main__":
    SegmentationToyDataset()
