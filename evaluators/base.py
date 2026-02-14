from abc import ABC, abstractmethod
from typing import Dict


class EvaluatorBase(ABC):
    """Base evaluator interface."""

    @abstractmethod
    def evaluate(self, pred_df, label: str = "run") -> Dict:
        raise NotImplementedError

    @abstractmethod
    def summary(self, args, args_dynamic, eval_summary: Dict):
        raise NotImplementedError

    def save_metrics(self, output_dir: str, timestamp: str, eval_summary: Dict) -> None:
        pass
