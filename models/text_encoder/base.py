from abc import ABC, abstractmethod


class TextEncoderBase(ABC):
    """Base text encoder interface."""

    @abstractmethod
    def encode_text(self, prompts):
        """Return normalized text features as a torch.Tensor."""
        raise NotImplementedError
