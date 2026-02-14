import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModel


@dataclass
class JinaComponents:
    model: object
    device: torch.device

    def encode_text(self, texts, task: str = "retrieval", prompt_name: Optional[str] = "query"):
        with torch.no_grad():
            outputs = self.model.encode_text(texts=texts, task=task, prompt_name=prompt_name)
        return outputs

    def encode_image(self, images, task: str = "retrieval"):
        with torch.no_grad():
            outputs = self.model.encode_image(images=images, task=task)
        return outputs


def build_jina_components(args, device) -> JinaComponents:
    model_id = getattr(args, "model", None) or "jinaai/jina-embeddings-v4"
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logging.info("Loading Jina embeddings model: %s", model_id)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return JinaComponents(model=model, device=device)
