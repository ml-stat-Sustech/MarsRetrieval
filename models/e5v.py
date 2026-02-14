import logging
from dataclasses import dataclass

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

LLAMA3_TEMPLATE = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
DEFAULT_IMAGE_INSTRUCTION = "<image>\nSummary above image in one word: "
DEFAULT_TEXT_INSTRUCTION = "<sent>\nSummary above sentence in one word: "


@dataclass
class E5VComponents:
    model: LlavaNextForConditionalGeneration
    processor: LlavaNextProcessor
    device: torch.device
    image_prompt: str
    text_prompt_template: str

    def build_text_prompts(self, texts):
        return [self.text_prompt_template.replace("<sent>", text) for text in texts]


def build_e5v_components(args, device) -> E5VComponents:
    model_id = getattr(args, "model", None) or "royokong/e5-v"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logging.info("Loading e5-V model: %s", model_id)
    processor = LlavaNextProcessor.from_pretrained(model_id)
    processor.patch_size = args.patch_size if hasattr(args, "patch_size") and args.patch_size is not None else 14
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # Some processor configs may miss patch_size; set a sensible default to avoid NoneType errors.
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and getattr(image_processor, "patch_size", None) is None:
        fallback = getattr(getattr(image_processor, "config", None), "patch_size", None) or 14
        image_processor.patch_size = fallback

    model.to(device)
    model.eval()

    image_prompt = LLAMA3_TEMPLATE.format(DEFAULT_IMAGE_INSTRUCTION)
    text_prompt_template = LLAMA3_TEMPLATE.format(DEFAULT_TEXT_INSTRUCTION)

    return E5VComponents(
        model=model,
        processor=processor,
        device=device,
        image_prompt=image_prompt,
        text_prompt_template=text_prompt_template,
    )
