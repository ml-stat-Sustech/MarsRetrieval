import torch
from torch.nn import functional as F

from ..jina import JinaComponents
from .base import TextEncoderBase


class JinaTextEncoder(TextEncoderBase):
    def __init__(self, components: JinaComponents):
        self.components = components

    def encode_text(self, prompts):
        with torch.no_grad():
            prompt_name = getattr(self.components, "prompt_name", "query")
            feats = self.components.encode_text(prompts, task="retrieval", prompt_name=prompt_name)
        # list[tensor] or tensor
        if isinstance(feats, list):
            feats = torch.stack(feats, dim=0)
        if torch.is_tensor(feats) and feats.dim() == 3:
            feats = feats.mean(dim=1)
        if not torch.is_tensor(feats):
            raise TypeError(f"Unsupported text feature type from Jina encode_text: {type(feats)}")
        feats = F.normalize(feats, p=2, dim=-1)
        return feats
