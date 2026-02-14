import torch
from torch.nn import functional as F

from ..aimv2_vl import AimV2VLComponents
from .base import ImageEncoderBase


class AimV2VLImageEncoder(ImageEncoderBase):
    """Vision-language image encoder for apple/aimv2-large-patch14-224-lit."""

    def __init__(self, components: AimV2VLComponents):
        processor = components.processor

        def _preprocess(img):
            return processor(images=img, return_tensors="pt")["pixel_values"][0]

        super().__init__(preprocess=_preprocess, device=components.device)
        self.components = components

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = images.to(self.device, dtype=torch.float16 if self.device.type == "cuda" else images.dtype)
        model = self.components.model
        with torch.no_grad():
            enc_out = model.image_encoder(pixel_values)
            img_vec = enc_out.last_hidden_state
            feats = model.image_projector(img_vec)
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
