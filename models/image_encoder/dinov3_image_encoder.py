import logging

import torch
from torch.nn import functional as F

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:  # pragma: no cover - older transformers
    from transformers import AutoProcessor as AutoImageProcessor  # type: ignore
    from transformers import AutoModel  # type: ignore

from .base import ImageEncoderBase


class DinoV3ImageEncoder(ImageEncoderBase):
    """DINOv3 image encoder via HuggingFace transformers."""

    def __init__(self, model_id: str, device: torch.device, pooling: str = "cls"):
        self.device = device
        self.pooling = pooling.lower()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval().to(device)

        # Preprocess returns a tensor suitable for DataLoader collation
        def _hf_preprocess(img):
            px = self.processor(images=img, return_tensors="pt")["pixel_values"]
            return px[0]

        self.preprocess = _hf_preprocess

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            if hasattr(outputs, "last_hidden_state"):
                token_feats = outputs.last_hidden_state
                if self.pooling == "mean":
                    feats = token_feats.mean(dim=1)
                else:
                    feats = token_feats[:, 0]
            elif hasattr(outputs, "pooler_output"):
                feats = outputs.pooler_output
            else:
                feats = outputs[0][:, 0]
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
