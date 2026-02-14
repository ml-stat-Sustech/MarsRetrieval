import torch
from torch.nn import functional as F

from ..gme import GMEComponents
from .base import TextEncoderBase


class GMETextEncoder(TextEncoderBase):
    """Text encoder wrapper for Alibaba GME (Qwen2-VL) embeddings."""

    def __init__(self, components: GMEComponents):
        self.components = components
        self.device = components.device

    def encode_text(self, prompts):
        feats = self.components.encode_text(prompts)
        if isinstance(feats, list):
            feats = torch.stack(feats, dim=0)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, device=self.device)
        feats = F.normalize(feats, p=2, dim=-1)
        return feats
