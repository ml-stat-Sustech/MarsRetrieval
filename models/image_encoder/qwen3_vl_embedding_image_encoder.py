from typing import List, Tuple, Union

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from ..qwen3_vl_embedding import Qwen3VLEmbeddingComponents
from .base import ImageEncoderBase


def qwen3_vl_embedding_collate(batch: List[Tuple[Union[str, Image.Image], str]]):
    images, names = zip(*batch)
    return list(images), list(names)


class Qwen3VLEmbeddingImageEncoder(ImageEncoderBase):
    """Image encoder wrapper for Qwen3-VL embedding."""

    def __init__(self, components: Qwen3VLEmbeddingComponents):
        super().__init__(preprocess=None, device=components.device)
        self.components = components
        self.collate_fn = qwen3_vl_embedding_collate
        self.use_path_inputs = True

    def _prepare_images(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        processed = []
        for img in images:
            if isinstance(img, str):
                processed.append(img)
                continue
            if torch.is_tensor(img):
                img = to_pil_image(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            processed.append(img)
        return processed

    def encode_image(self, images):
        images = self._prepare_images(images)
        inputs = [{"image": img} for img in images]
        with torch.no_grad():
            feats = self.components.model.process(inputs)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats)
        return feats.to(dtype=torch.float32, device=self.components.device)
