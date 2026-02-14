import logging
from typing import List

import numpy as np
import torch
from torch.nn import functional as F


PROMPT_TEMPLATES = [
    "a photo of {}, a type of martian terrain",
    "a satellite photo of {}.",
    "a high-resolution remote sensing image of {} on Mars.",
]


def _format_concept(name: str) -> str:
    return name.replace("_", " ").replace("-", " ")


def build_landform_query(args, text_encoder, query_mode: str):
    if query_mode != "text":
        raise ValueError("Landform retrieval only supports text query_mode.")
    if text_encoder is None:
        raise ValueError("Text encoder is required for landform retrieval.")

    class_names: List[str] = getattr(args, "landform_classes", None) or []
    if not class_names:
        raise ValueError("No landform classes found; ensure dataset build ran before querying.")

    queries = []
    query_names = []
    with torch.no_grad():
        for cls in class_names:
            phrase = _format_concept(cls)
            prompts = [template.format(phrase) for template in PROMPT_TEMPLATES]
            feats = text_encoder.encode_text(prompts)
            if len(prompts) > 1:
                query = feats.mean(dim=0, keepdim=True)
                query = F.normalize(query, p=2, dim=-1)
            else:
                query = feats
            queries.append(query)
            query_names.append(cls)

    query_tensor = torch.cat(queries, dim=0)
    logging.info("Built %s landform text queries", len(query_names))
    return query_tensor.cpu().numpy(), query_names
