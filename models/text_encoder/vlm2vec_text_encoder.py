from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

# Ensure third_party/vlm2vec is importable for src.*
_VLM2VEC_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "vlm2vec"
if _VLM2VEC_ROOT.exists() and str(_VLM2VEC_ROOT) not in sys.path:
    sys.path.append(str(_VLM2VEC_ROOT))

from src.utils.basic_utils import batch_to_device

from ..vlm2vec import VLM2VecComponents
from .base import TextEncoderBase


class VLM2VecTextEncoder(TextEncoderBase):
    """Text encoder wrapper for VLM2Vec (Qwen2-VL backbone)."""

    def __init__(self, components: VLM2VecComponents):
        self.components = components
        self.device = components.device

    def encode_text(self, prompts):
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]

        model_inputs = {
            "text": list(prompts),
            "images": [None] * len(prompts),
        }
        inputs = self.components.process_fn(model_inputs, self.components.processor)
        inputs = batch_to_device(inputs, self.device)

        with torch.no_grad():
            autocast_ctx = (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                feats = self.components.model(tgt=inputs)["tgt_reps"]

        if not self.components.model.normalize:
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
