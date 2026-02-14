import logging
from dataclasses import dataclass

import torch
from transformers import AutoModel


@dataclass
class BGEVLComponents:
    model: object
    device: torch.device

    def encode_text(self, texts, **kwargs):
        processor = getattr(self.model, "processor", None)
        if processor is None:
            return self.model.encode(text=texts, **kwargs)
        config = getattr(self.model, "config", None)
        text_config = getattr(config, "text_config", None)
        max_len = getattr(text_config, "max_position_embeddings", None)
        if max_len is None:
            max_len = getattr(config, "max_position_embeddings", 77)
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(self.device)
        return self.model.encode_text(text_inputs)

    def encode_image(self, images, **kwargs):
        return self.model.encode(images=images, **kwargs)


def build_bgevl_components(args, device) -> BGEVLComponents:
    model_id = getattr(args, "model", None) or "BAAI/BGE-VL-base"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logging.info("Loading BGE-VL model: %s", model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
    # BGE-VL uses its own processor via set_processor
    model.set_processor(model_id)
    model.to(device)
    model.eval()
    return BGEVLComponents(model=model, device=device)
