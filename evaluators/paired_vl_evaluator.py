import logging
from typing import Dict

import numpy as np

from .base import EvaluatorBase


def _compute_ranks(similarity: np.ndarray) -> np.ndarray:
    # similarity: [N, N], higher is better
    order = np.argsort(-similarity, axis=1)
    ranks = np.empty(similarity.shape[0], dtype=np.int32)
    for i in range(similarity.shape[0]):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    return ranks


class PairedVLEvaluator(EvaluatorBase):
    def evaluate(self, pred_df, label: str = "run") -> Dict:
        eval_input = None
        if isinstance(pred_df, dict):
            if "__eval_input__" in pred_df:
                eval_input = pred_df["__eval_input__"] or {}
            elif "image_features" in pred_df and "text_features" in pred_df:
                eval_input = pred_df
        if not eval_input:
            raise ValueError("Cross-modal evaluation requires image/text features input.")

        image_features = eval_input["image_features"]
        text_features = eval_input["text_features"]

        sim_i2t = image_features @ text_features.T
        ranks_i2t = _compute_ranks(sim_i2t)
        sim_t2i = sim_i2t.T
        ranks_t2i = _compute_ranks(sim_t2i)

        metrics = {
            "i2t_r1": float(np.mean(ranks_i2t <= 1)),
            "i2t_r10": float(np.mean(ranks_i2t <= 10)),
            "i2t_medr": float(np.median(ranks_i2t)),
            "i2t_mrr": float(np.mean(1.0 / ranks_i2t)),
            "t2i_r1": float(np.mean(ranks_t2i <= 1)),
            "t2i_r10": float(np.mean(ranks_t2i <= 10)),
            "t2i_medr": float(np.median(ranks_t2i)),
            "t2i_mrr": float(np.mean(1.0 / ranks_t2i)),
        }
        logging.info(
            "[%s] I2T R@1=%.4f R@10=%.4f MedR=%.0f MRR=%.4f | "
            "T2I R@1=%.4f R@10=%.4f MedR=%.0f MRR=%.4f",
            label,
            metrics["i2t_r1"],
            metrics["i2t_r10"],
            metrics["i2t_medr"],
            metrics["i2t_mrr"] * 100,
            metrics["t2i_r1"],
            metrics["t2i_r10"],
            metrics["t2i_medr"],
            metrics["t2i_mrr"] * 100,
        )
        return metrics

    def summary(self, args, args_dynamic, eval_summary: Dict):
        headers = [
            "model_name",
            "pretrained",
            "resume_post_train",
            "caption_key",
            "i2t_r1",
            "i2t_r10",
            "i2t_mrr",
            "i2t_medr",
            "t2i_r1",
            "t2i_r10",
            "t2i_mrr",
            "t2i_medr",
        ]
        row = [
            args.model,
            args.pretrained,
            args.resume_post_train or "",
            getattr(args, "caption_key", "") or "",
            round(eval_summary.get("i2t_r1", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("i2t_r10", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("i2t_mrr", 0.0) * 100, 2) if eval_summary else "",
            int(round(eval_summary.get("i2t_medr", 0.0), 0)) if eval_summary else "",
            round(eval_summary.get("t2i_r1", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("t2i_r10", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("t2i_mrr", 0.0) * 100, 2) if eval_summary else "",
            int(round(eval_summary.get("t2i_medr", 0.0), 0)) if eval_summary else "",
        ]
        return headers, row
