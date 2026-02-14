import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Ensure third_party/ops_mm_embedding is importable
_OPS_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ops_mm_embedding"
if _OPS_ROOT.exists() and str(_OPS_ROOT) not in sys.path:
    sys.path.append(str(_OPS_ROOT))

from ops_mm_embedding_v1 import OpsMMEmbeddingV1  # type: ignore


@dataclass
class OpsMMV1Components:
    model: OpsMMEmbeddingV1
    device: torch.device


def build_opsmm_v1_components(args, device) -> OpsMMV1Components:
    model_id = getattr(args, "model", None) or "OpenSearch-AI/Ops-MM-embedding-v1-2B"
    attn_impl = getattr(args, "attn_implementation", None) or "flash_attention_2"

    logging.info("Loading Ops-MM v1 model: %s (attn=%s)", model_id, attn_impl)
    model = OpsMMEmbeddingV1(
        model_name=model_id,
        device=str(device),
        attn_implementation=attn_impl,
    )
    model.eval()
    return OpsMMV1Components(model=model, device=device)
