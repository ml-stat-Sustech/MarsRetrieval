import torch
from io import BytesIO
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image

from ..bgevl import BGEVLComponents
from .base import ImageEncoderBase


def bgevl_collate(batch):
    images, names = zip(*batch)
    return list(images), list(names)


class BGEVLImageEncoder(ImageEncoderBase):
    def __init__(self, components: BGEVLComponents):
        super().__init__(preprocess=None, device=components.device)
        self.components = components
        self.collate_fn = bgevl_collate
        # signal dataset/query loaders to pass file paths instead of PIL images
        self.use_path_inputs = True

    def encode_image(self, images):
        # Expect list/tuple of file paths (strings); fall back to BytesIO if PIL/tensor is passed.
        if not isinstance(images, (list, tuple)):
            images = [images]
        processed = []
        tmp_buffers = []
        for img in images:
            if isinstance(img, (str, bytes)):
                processed.append(img)
            elif torch.is_tensor(img):
                img = to_pil_image(img)
                buf = BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=95)
                buf.seek(0)
                processed.append(buf)
                tmp_buffers.append(buf)
            elif isinstance(img, Image.Image):
                buf = BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=95)
                buf.seek(0)
                processed.append(buf)
                tmp_buffers.append(buf)
            elif hasattr(img, "read"):
                processed.append(img)
            else:
                processed.append(img)

        with torch.no_grad():
            feats = self.components.encode_image(images=processed)

        for buf in tmp_buffers:
            try:
                buf.close()
            except Exception:
                pass

        if isinstance(feats, list):
            feats = torch.stack(feats, dim=0)
        if not torch.is_tensor(feats):
            raise TypeError(f"Unsupported image feature type from BGE-VL encode_image: {type(feats)}")

        feats = F.normalize(feats, p=2, dim=-1)
        return feats
