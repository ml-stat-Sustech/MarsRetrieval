import logging
from dataclasses import dataclass

import torch
from transformers import AutoImageProcessor, AutoModel


@dataclass
class AimV2VisComponents:
    model: object
    processor: object
    device: torch.device


def build_aimv2_vis_components(args, device) -> AimV2VisComponents:
    model_id = getattr(args, "model", None) or "apple/aimv2-large-patch14-448"
    revision = getattr(args, "pretrained", None) or getattr(args, "revision", None) or "cefb13f21003bdadba65bfbee956c82b976cd23d"
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logging.info("Loading AIMv2 vision model: %s (rev=%s)", model_id, revision)
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return AimV2VisComponents(model=model, processor=processor, device=device)
