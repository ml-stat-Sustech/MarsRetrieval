import torch
from torch.nn import functional as F

from ..e5v import E5VComponents
from .base import TextEncoderBase


class E5VTextEncoder(TextEncoderBase):
    """Text encoder wrapper for e5-V (LLaVA-Next backbone)."""

    def __init__(self, components: E5VComponents):
        self.components = components
        self.device = components.device

    def encode_text(self, prompts):
        prompts = self.components.build_text_prompts(prompts)
        inputs = self.components.processor(
            text=prompts,
            images=None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.components.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            feats = outputs.hidden_states[-1][:, -1, :]
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
