from typing import List
import torch
from torch import nn

class LDMSeg(nn.Module):
    def __init__(self, 
                 image_vea: nn.Module,
                 diffusion_model: nn.Module,
                 latent_dec: nn.Module,
                 *args, **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self._image_vea = image_vea
        self._diffusion_model = diffusion_model
        self._latent_dec = latent_dec

    def encode_rgb(self, images: torch.Tensor) -> torch.Tensor:
        # TODO Check if dataloader already did the normalization
        images = 2. * images - 1.
        rgb_latents = self._image_vea.encode(images)
        return rgb_latents

    def diffusion_process(self, latent_image: torch.Tensor, latent_label: torch.Tensor, t: int) -> torch.Tensor:
        pass

    def decode_latent(self, latent_image, latent_label) -> torch.Tensor:
        pass


def segment_video_slow(model: nn.Module, frames: List[torch.Tensor]):
    labels = [model.predict(f) for f in frames]
    return labels

def segment_video_fast(model: nn.Module, frames: List[torch.Tensor]):
    # load model parts
    encode_inputs = model.encode_image # Encodes the model to latent space (Decoder part is not used during inference)
    vae_semseg = model # Decodes latent space to panoptic segmentation (Encoder part is not used during inference)
    unet = ... # Diffusion process

    first_frame, *frames = [...] # Load frames from a video

    # Predict initial label 'normally'
    image_f = vae_image(first_frame)
    init_noise = torch.random_like(image_f, 0, 1)

    for frame in frames:
        image_f = vae_image(frame)
        

def main():
    # load model parts
    vae_image = ... # Encodes the model to latent space (Decoder part is not used during inference)
    vae_semseg = ... # Decodes latent space to panoptic segmentation (Encoder part is not used during inference)
    unet = ... # Diffusion process
    
    frames = ... # Load frames from a video

if __name__ == "__main__":
    main()