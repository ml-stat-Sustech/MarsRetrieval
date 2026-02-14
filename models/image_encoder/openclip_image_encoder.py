import logging

import torch
from torch.nn import functional as F

from ..openclip import OpenCLIPComponents
from .base import ImageEncoderBase


class OpenCLIPImageEncoder(ImageEncoderBase):
    def __init__(self, components: OpenCLIPComponents, device: torch.device):
        super().__init__(preprocess=components.preprocess_val, device=device)
        self.model = components.model

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            feats = self.model.encode_image(images.to(self.device))
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
