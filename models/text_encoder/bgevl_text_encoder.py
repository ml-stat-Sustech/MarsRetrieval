import torch
from torch.nn import functional as F

from ..bgevl import BGEVLComponents
from .base import TextEncoderBase


class BGEVLTextEncoder(TextEncoderBase):
    def __init__(self, components: BGEVLComponents):
        self.components = components

    def encode_text(self, prompts):
        with torch.no_grad():
            feats = self.components.encode_text(texts=prompts)
        if isinstance(feats, list):
            feats = torch.stack(feats, dim=0)
        if not torch.is_tensor(feats):
            raise TypeError(f"Unsupported text feature type from BGE-VL encode_text: {type(feats)}")
        feats = F.normalize(feats, p=2, dim=-1)
        return feats
