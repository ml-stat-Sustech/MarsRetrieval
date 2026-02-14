import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Ensure third_party/qwen3_vl_embedding is importable
_QWEN3_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "qwen3_vl_embedding"
if _QWEN3_ROOT.exists() and str(_QWEN3_ROOT) not in sys.path:
    sys.path.append(str(_QWEN3_ROOT))

from scirpts import Qwen3VLEmbedder  # type: ignore


@dataclass
class Qwen3VLEmbeddingComponents:
    model: Qwen3VLEmbedder
    device: torch.device


def build_qwen3_vl_embedding_components(args, device) -> Qwen3VLEmbeddingComponents:
    model_id = getattr(args, "model", None) or "Qwen/Qwen3-VL-Embedding-2B"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    attn_impl = getattr(args, "attn_implementation", None)

    logging.info("Loading Qwen3-VL embedding model: %s", model_id)
    kwargs = {"torch_dtype": dtype}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = Qwen3VLEmbedder(model_name_or_path=model_id, **kwargs)
    return Qwen3VLEmbeddingComponents(model=model, device=device)
