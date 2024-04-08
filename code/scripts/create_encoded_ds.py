"""Encodes a dataset into a feature space of a model

Taking a feature extractor `f(x)->h` (Usually the encoder of a model)
This converts the (x, y) pairs into (h, y) pairs.
"""

import os

from diffusers import AutoencoderKL
import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transforms
import tqdm

from datasets.coco import CoCoDataset

BATCH_SIZE = 32


def main(model, root_dir, split):
    save_dir = os.path.join(root_dir, f"{split}_latents")
    os.makedirs(save_dir, exist_ok=True)

    image_net_transforms = [
        # Rescale to [0, 1], then normalize using mean and std of ImageNet1K DS
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
    data_transforms = transforms.Compose(
        [transforms.Resize((128, 256)), *image_net_transforms]
    )
    dataset = CoCoDataset(split, transform=data_transforms)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=4,
    )
    print(f"Starting on split: {split}")
    create_ds(model, dataloader, save_dir)
    print(f"Finished split: {split}")


@torch.no_grad
def create_ds(model, dataloader, save_dir):
    model = model.to(device="cuda")
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        parameters = model.encode(
            batch["img"].to(device="cuda", non_blocking=True)
        ).latent_dist.parameters.cpu()
        save_batch(batch_idx, parameters, save_dir)


def save_batch(batch_idx: int, parameters: torch.Tensor, save_dir: str):
    for i, x in enumerate(parameters):
        idx = batch_idx * BATCH_SIZE + i
        path = os.path.join(save_dir, f"vae_latent_{idx}.pt")
        torch.save(x.clone(), path)


if __name__ == "__main__":
    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    model = AutoencoderKL.from_single_file(url)
    model.eval()

    data_dir = "/datasets/coco/"

    for split in ["train", "val"]:
        main(model, data_dir, split)
