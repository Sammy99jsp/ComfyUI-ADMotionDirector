from dataclasses import dataclass
from email import utils

import torch

@dataclass
class LabeledVid:
    """
    Single Video with prompt attached.
    """
    frames: torch.Tensor
    prompt: str

    def get_frames(self) -> torch.Tensor:
        return self.frames


@dataclass
class TrainingVideo:
    encoder_hidden_states: torch.Tensor
    pixels: torch.Tensor
    latent: torch.Tensor

    @classmethod
    def new(cls, encoder_hidden_states, pixels, latent):
        return cls(encoder_hidden_states=encoder_hidden_states, pixels=pixels, latent=latent)