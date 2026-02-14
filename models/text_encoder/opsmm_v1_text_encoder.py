import torch

from ..opsmm_v1 import OpsMMV1Components
from .base import TextEncoderBase


class OpsMMV1TextEncoder(TextEncoderBase):
    """Text encoder wrapper for Ops-MM v1 embedding."""

    def __init__(self, components: OpsMMV1Components):
        self.components = components
        self.device = components.device

    def encode_text(self, prompts):
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        with torch.no_grad():
            feats = self.components.model.get_text_embeddings(list(prompts))
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats)
        return feats.to(dtype=torch.float32, device=self.device)
