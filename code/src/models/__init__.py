from typing import Optional, Callable
import torch


class ModelOutput:
    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self._logits = logits
        self._out = out
        self._act_fn = act_fn

    @property
    def logits(self) -> Optional[torch.Tensor]:
        return self._logits

    @property
    def out(self) -> torch.Tensor:
        if self._out is None:
            self._out = self._act_fn(self.logits)
        return self._out


from models.u_net import UNet
from models.vae import VAE

__all__ = ["UNet", "VAE"]
