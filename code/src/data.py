from typing import Any
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import v2 as tranforms
from torchvision.tv_tensors import Mask

class SegmentationToyDataset(data.Dataset):
    """A toy dataset for image segmentation
    
    """
    def __init__(self):
        self.images = []

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

        img[0, x-self._x_size:x+self._x_size, y-self._y_size:y+self._y_size] = 255
        mask[0, x-self._x_size:x+self._x_size, y-self._y_size:y+self._y_size] = 255

        return img, mask


if __name__ == "__main__":
    SegmentationToyDataset()