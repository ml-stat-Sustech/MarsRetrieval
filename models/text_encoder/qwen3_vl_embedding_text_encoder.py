import torch

from ..qwen3_vl_embedding import Qwen3VLEmbeddingComponents
from .base import TextEncoderBase


class Qwen3VLEmbeddingTextEncoder(TextEncoderBase):
    """Text encoder wrapper for Qwen3-VL embedding."""

    def __init__(self, components: Qwen3VLEmbeddingComponents):
        self.components = components
        self.device = components.device

    def encode_text(self, prompts):
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        inputs = [{"text": text} for text in prompts]
        with torch.no_grad():
            feats = self.components.model.process(inputs)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats)
        return feats.to(dtype=torch.float32, device=self.device)
