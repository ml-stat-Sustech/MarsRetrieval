import torch
from torch.nn import functional as F

from ..aimv2_vl import AimV2VLComponents
from .base import TextEncoderBase


class AimV2VLTextEncoder(TextEncoderBase):
    """Vision-language text encoder for apple/aimv2-large-patch14-224-lit."""

    def __init__(self, components: AimV2VLComponents):
        self.components = components
        self.processor = components.processor
        self.device = components.device

    def encode_text(self, prompts):
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]

        inputs = self.processor(
            text=list(prompts),
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            model = self.components.model
            txt_out = model.text_encoder(inputs.input_ids)
            txt_vec = txt_out.last_hidden_state
            feats = model.text_projector(txt_vec)
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
