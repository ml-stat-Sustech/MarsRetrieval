from typing import List, Tuple, Union

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from ..opsmm_v1 import OpsMMV1Components
from .base import ImageEncoderBase


def opsmm_v1_collate(batch: List[Tuple[Union[str, Image.Image], str]]):
    images, names = zip(*batch)
    return list(images), list(names)


class OpsMMV1ImageEncoder(ImageEncoderBase):
    """Image encoder wrapper for Ops-MM v1 embedding."""

    def __init__(self, components: OpsMMV1Components):
        super().__init__(preprocess=None, device=components.device)
        self.components = components
        self.collate_fn = opsmm_v1_collate
        self.use_path_inputs = True

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
        with torch.no_grad():
            feats = self.components.model.get_image_embeddings(images)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats)
        return feats.to(torch.float32)
