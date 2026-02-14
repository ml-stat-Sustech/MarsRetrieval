from abc import ABC, abstractmethod
from typing import Any, Dict


class DatasetBuilderBase(ABC):
    """Base class for building retrieval databases."""

    @abstractmethod
    def build(self, args, image_encoder, delta: float) -> Dict[str, Any]:
        raise NotImplementedError
