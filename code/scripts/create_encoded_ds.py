"""Encodes a dataset into a feature space of a model

Taking a feature extractor `f(x)->h` (Usually the encoder of a model)
This converts the (x, y) pairs into (h, y) pairs.
"""
import os
import asyncio

from diffusers import AutoencoderKL
import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transforms
import tqdm

from datasets.coco import CoCoDataset

DATA_ROOT = "/datasets/coco/latents_2/"
BATCH_SIZE = 32


def process_batch(model, batch_idx, image):
    model_out = model.encode(image.to(device="cuda", non_blocking=True))
    return batch_idx, model_out.latent_dist.parameters

@torch.no_grad
def create_ds(model, dataloader):
    model = model.to(device="cuda")
    for batch_idx, (image, _) in enumerate(tqdm.tqdm(dataloader)):
        parameters = model.encode(image.to(device="cuda", non_blocking=True)).latent_dist.parameters.cpu()
        save_batch(batch_idx, parameters)

def save_batch(batch_idx: int, parameters: torch.Tensor):
    for i, x in enumerate(parameters):
        idx = batch_idx * BATCH_SIZE + i
        path = os.path.join(DATA_ROOT, f"{split}_vae_latent_{idx}.pt")
        torch.save({"latent_parameters": x, "target": idx}, path)


if __name__ == "__main__":
    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    model = AutoencoderKL.from_single_file(url)

    split = "train"
    image_net_transforms = [
        # Rescale to [0, 1], then normalize using mean and std of ImageNet1K DS
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    data_transforms = transforms.Compose(
        [transforms.Resize((128, 128)), *image_net_transforms]
    )
    dataset = CoCoDataset(split, transform=data_transforms)
    
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4
    )

    os.makedirs(DATA_ROOT, exist_ok=True)
    
    create_ds(model, dataloader)
