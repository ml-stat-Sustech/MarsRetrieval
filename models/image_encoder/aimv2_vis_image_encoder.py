import torch
from torch.nn import functional as F

from ..aimv2_vis import AimV2VisComponents
from .base import ImageEncoderBase


class AimV2VisImageEncoder(ImageEncoderBase):
    """Pure vision encoder for apple/aimv2-large-patch14-448."""

    def __init__(self, components: AimV2VisComponents):
        processor = components.processor

        def _preprocess(img):
            return processor(images=img, return_tensors="pt")["pixel_values"][0]

        super().__init__(preprocess=_preprocess, device=components.device)
        self.components = components

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = images.to(self.device, dtype=torch.float16 if self.device.type == "cuda" else images.dtype)
        with torch.no_grad():
            outputs = self.components.model(pixel_values=pixel_values)
            # Pool token features (mean over sequence)
            feats = outputs.last_hidden_state.mean(dim=1)
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
