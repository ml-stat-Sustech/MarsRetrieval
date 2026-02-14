from typing import List, Tuple

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image

from ..e5v import E5VComponents
from .base import ImageEncoderBase


def e5v_collate(batch: List[Tuple[Image.Image, str]]):
    images, names = zip(*batch)
    return list(images), list(names)


class E5VImageEncoder(ImageEncoderBase):
    """Image encoder wrapper for e5-V (LLaVA-Next backbone)."""

    def __init__(self, components: E5VComponents):
        super().__init__(preprocess=None, device=components.device)
        self.components = components
        self.collate_fn = e5v_collate

    def _prepare_images(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        processed = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif torch.is_tensor(img):
                img = to_pil_image(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            processed.append(img)
        return processed

    def encode_image(self, images):
        images = self._prepare_images(images)
        prompts = [self.components.image_prompt] * len(images)
        inputs = self.components.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.components.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            feats = outputs.hidden_states[-1][:, -1, :]
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
