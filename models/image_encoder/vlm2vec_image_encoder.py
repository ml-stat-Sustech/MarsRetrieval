from contextlib import nullcontext
from pathlib import Path
import sys
from typing import List, Tuple

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image

# Ensure third_party/vlm2vec is importable for src.*
_VLM2VEC_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "vlm2vec"
if _VLM2VEC_ROOT.exists() and str(_VLM2VEC_ROOT) not in sys.path:
    sys.path.append(str(_VLM2VEC_ROOT))

from src.utils.basic_utils import batch_to_device

from ..vlm2vec import VLM2VecComponents
from .base import ImageEncoderBase


def vlm2vec_collate(batch: List[Tuple[Image.Image, str]]):
    images, names = zip(*batch)
    return list(images), list(names)


class VLM2VecImageEncoder(ImageEncoderBase):
    """Image encoder wrapper for VLM2Vec (Qwen2-VL backbone)."""

    def __init__(self, components: VLM2VecComponents):
        super().__init__(preprocess=None, device=components.device)
        self.components = components
        self.collate_fn = vlm2vec_collate

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
        model_inputs = {
            "text": [self.components.image_prompt] * len(images),
            "images": images,
        }
        inputs = self.components.process_fn(model_inputs, self.components.processor)
        inputs = batch_to_device(inputs, self.device)

        with torch.no_grad():
            autocast_ctx = (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                feats = self.components.model(qry=inputs)["qry_reps"]

        if not self.components.model.normalize:
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
