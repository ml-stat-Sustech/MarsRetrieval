import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

# Ensure third_party/vlm2vec is on the import path for src.*
_VLM2VEC_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "vlm2vec"
if _VLM2VEC_ROOT.exists() and str(_VLM2VEC_ROOT) not in sys.path:
    sys.path.append(str(_VLM2VEC_ROOT))

from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import QWEN2_VL, Qwen2_VL_process_fn, VLM_IMAGE_TOKENS, load_processor
from src.utils.basic_utils import batch_to_device


@dataclass
class VLM2VecComponents:
    model: MMEBModel
    processor: object
    device: torch.device
    model_args: ModelArguments
    process_fn: Callable
    image_prompt: str


def build_vlm2vec_components(args, device) -> VLM2VecComponents:
    # Match upstream demo defaults
    base_model = getattr(args, "model", None) or "VLM2Vec/VLM2Vec-V2.0"
    # Optional fine-tuned checkpoint; defaults to model weights if not provided.

    model_args = ModelArguments(
        model_name=base_model,
        pooling="last",
        normalize=True,
        model_backbone="qwen2_vl",
        lora=True,
    )
    data_args = DataArguments()

    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device, dtype=dtype)
    model.eval()

    image_prompt = (
        f"{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: "
        "What is in the image"
    )

    return VLM2VecComponents(
        model=model,
        processor=processor,
        device=device,
        model_args=model_args,
        process_fn=Qwen2_VL_process_fn,
        image_prompt=image_prompt,
    )
