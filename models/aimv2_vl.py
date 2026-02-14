import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModel, AutoProcessor


@dataclass
class AimV2VLComponents:
    model: object
    processor: object
    device: torch.device


def build_aimv2_vl_components(args, device) -> AimV2VLComponents:
    model_id = getattr(args, "model", None) or "apple/aimv2-large-patch14-224-lit"
    revision = getattr(args, "pretrained", None) or getattr(args, "revision", None) or "c2cd59a786c4c06f39d199c50d08cc2eab9f8605"
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logging.info("Loading AIMv2 model: %s (rev=%s)", model_id, revision)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return AimV2VLComponents(model=model, processor=processor, device=device)
