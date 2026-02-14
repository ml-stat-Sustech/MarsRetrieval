import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image

from ..jina import JinaComponents
from .base import ImageEncoderBase


def jina_collate(batch):
    images, names = zip(*batch)
    return list(images), list(names)


class JinaImageEncoder(ImageEncoderBase):
    def __init__(self, components: JinaComponents):
        # No preprocessing; the Jina model handles it internally.
        super().__init__(preprocess=None, device=components.device)
        self.components = components
        self.collate_fn = jina_collate

    def encode_image(self, images):
        # images: list of PIL.Image (from custom collate) or a torch Tensor batch
        if torch.is_tensor(images):
            images = [to_pil_image(img) for img in images]
        with torch.no_grad():
            feats = self.components.encode_image(images, task="retrieval")
        # Jina returns: list[tensor] (one per image) or tensor
        if isinstance(feats, list):
            feats = torch.stack(feats, dim=0)
        if not torch.is_tensor(feats):
            raise TypeError(f"Unsupported image feature type from Jina encode_image: {type(feats)}")
        feats = F.normalize(feats, p=2, dim=-1)
        return feats
