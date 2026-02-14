from typing import List, Tuple

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image

from ..gme import GMEComponents
from .base import ImageEncoderBase


def gme_collate(batch: List[Tuple[Image.Image, str]]):
    images, names = zip(*batch)
    return list(images), list(names)


class GMEImageEncoder(ImageEncoderBase):
    """Image encoder wrapper for Alibaba GME (Qwen2-VL) embeddings."""

    def __init__(self, components: GMEComponents):
        super().__init__(preprocess=None, device=components.device)
        self.components = components
        self.collate_fn = gme_collate

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
            else:
                raise TypeError(f"Unsupported image type for GME encoder: {type(img)}")
            processed.append(img)
        return processed

    def encode_image(self, images, is_query: bool = False):
        images = self._prepare_images(images)
        feats = self.components.encode_image(images=images, is_query=is_query)
        if isinstance(feats, list):
            feats = torch.stack(feats, dim=0)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, device=self.device)
        feats = F.normalize(feats, p=2, dim=-1)
        return feats
