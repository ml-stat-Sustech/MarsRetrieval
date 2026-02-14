from abc import ABC, abstractmethod
from typing import Any


class ImageEncoderBase(ABC):
    """Base image encoder interface."""

    def __init__(self, preprocess: Any, device):
        self.preprocess = preprocess
        self.device = device

    @abstractmethod
    def encode_image(self, images):
        """Return normalized image features as a torch.Tensor on CPU or GPU."""
        raise NotImplementedError

    def get_processor(self):
        """Return the preprocessing callable used for images."""
        return self.preprocess
