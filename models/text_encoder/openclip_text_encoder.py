import torch
from torch.nn import functional as F

from ..openclip import OpenCLIPComponents
from .base import TextEncoderBase


class OpenCLIPTextEncoder(TextEncoderBase):
    def __init__(self, components: OpenCLIPComponents, device: torch.device):
        self.model = components.model
        self.tokenizer = components.tokenizer
        self.device = device

    def encode_text(self, prompts):
        self.model.eval()
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
