from typing import Tuple

from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torchvision
import hydra
import os

from .utils import LatentTensor
from .coco import CoCoDataset
from .fiftyone import FiftyOneDataset
from .oxford_pet import OxfordPetDataset, OxfordPetForegroundDataset
from .toy_data import ToySegmentationTransform


def create_dataloaders(
    cfg: DictConfig,
    data_transforms: torchvision.transforms.v2.Transform,
) -> Tuple[DictConfig, DataLoader, DataLoader]:
    """Create the Train and Validation dataset based on an hydra DictConfig

    Args:
        cfg (DictConfig): The config dict describing the Dataset
        data_transforms (torchvision.transforms.v2): A torchvision transform to be applied to each sample individually.

    Returns:
        Tuple[DictConfig, DataLoader, DataLoader]: (
            modified_cfg: The config with additional parameters,
            train_dataloader: The dataloader to be used for training,
            val_dataloader: The dataloader to be used for validation,
        )
    """
    dataset_factory = hydra.utils.instantiate(
        cfg.dataset, _partial_=True
    )
    train_dataset = dataset_factory(
        split="train", transform=data_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ.get("SLURM_NTASKS", os.cpu_count())),
        pin_memory=True,
    )
    val_dataset = dataset_factory(
        split="val", transform=data_transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ.get("SLURM_NTASKS", os.cpu_count())),
        pin_memory=True,
    )

    # Set some 'global' information
    cfg.class_weights = train_dataset.class_weights
    cfg.class_map = train_dataset.class_map
    cfg.num_classes = len(cfg.class_map)
    cfg.ignore_index = train_dataset.ignore_index

    return cfg, train_loader, val_loader


__all__ = [
    "LatentTensor",
    "CoCoDataset",
    "FiftyOneDataset",
    "OxfordPetDataset",
    "OxfordPetForegroundDataset",
    "ToySegmentationTransform",
    "create_dataloaders",
]
