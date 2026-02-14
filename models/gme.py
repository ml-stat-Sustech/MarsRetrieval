import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModel
from transformers.utils.versions import require_version

DEFAULT_TEXT_INSTRUCTION = "Find an image that matches the given text."


@dataclass
class GMEComponents:
    model: object
    device: torch.device
    text_instruction: str = DEFAULT_TEXT_INSTRUCTION

    def encode_text(self, texts, instruction: Optional[str] = None):
        prompt = instruction or self.text_instruction
        with torch.no_grad():
            return self.model.get_text_embeddings(texts=texts, instruction=prompt)

    def encode_image(self, images, is_query: bool = False):
        with torch.no_grad():
            return self.model.get_image_embeddings(images=images, is_query=is_query)


def build_gme_components(args, device) -> GMEComponents:
    require_version(
        "transformers<4.52.0",
        "The GME remote code has known issues with transformers>=4.52.0; please use transformers==4.51.3",
    )
    model_id = getattr(args, "model", None) or "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logging.info("Loading GME model: %s", model_id)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    instruction = getattr(args, "gme_text_instruction", None) or DEFAULT_TEXT_INSTRUCTION
    return GMEComponents(model=model, device=device, text_instruction=instruction)
