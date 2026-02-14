import logging
from typing import Dict, List
import os
import csv

import numpy as np

from .base import EvaluatorBase


class LandformlEvaluator(EvaluatorBase):
    def __init__(self, gt_map: Dict[str, set], max_k: int = None):
        self.gt_map = gt_map or {}
        self.max_k = max_k

    def _compute_metrics(self, ranked: List[str], gt_set: set) -> Dict[str, float]:
        hits = 0
        precisions = []
        for i, img in enumerate(ranked, start=1):
            if img in gt_set:
                hits += 1
                precisions.append(hits / i)
        ap = sum(precisions) / len(gt_set) if gt_set else 0.0

        top10 = ranked[:10]
        hit10 = 1.0 if set(top10) & gt_set else 0.0
        r_n = len(gt_set)
        r_slice = ranked[:r_n] if r_n > 0 else []
        r_precision = (len(set(r_slice) & gt_set) / r_n) if r_n > 0 else 0.0
        dcg = 0.0
        for i, img in enumerate(top10, start=1):
            if img in gt_set:
                dcg += 1.0 / np.log2(i + 1)
        ideal_hits = min(len(gt_set), 10)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1)) if ideal_hits else 0.0
        ndcg10 = (dcg / idcg) if idcg > 0 else 0.0

        return {
            "mAP": ap,
            "hit@10": hit10,
            "r_precision": r_precision,
            "ndcg@10": ndcg10,
        }

    def evaluate(self, pred_df, label: str = "run") -> Dict:
        if pred_df.empty:
            logging.warning("Prediction dataframe is empty, skipping evaluation.")
            return {}

        query_names = sorted(set(pred_df["query_name"].tolist()))
        ap_list: List[float] = []
        hit10_list: List[float] = []
        r_precision_list: List[float] = []
        ndcg10_list: List[float] = []
        per_class: Dict[str, Dict[str, float]] = {}

        for q in query_names:
            gt_set = self.gt_map.get(q, set())
            if not gt_set:
                continue
            df_q = pred_df[pred_df["query_name"] == q].sort_values("similarity", ascending=False)
            if self.max_k is not None:
                df_q = df_q.head(self.max_k)
            ranked = df_q["image_name"].tolist()
            metrics = self._compute_metrics(ranked, gt_set)
            per_class[q] = metrics

            ap_list.append(metrics["mAP"])
            hit10_list.append(metrics["hit@10"])
            r_precision_list.append(metrics["r_precision"])
            ndcg10_list.append(metrics["ndcg@10"])

        metrics = {
            "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
            "hit@10": float(np.mean(hit10_list)) if hit10_list else 0.0,
            "r_precision": float(np.mean(r_precision_list)) if r_precision_list else 0.0,
            "ndcg@10": float(np.mean(ndcg10_list)) if ndcg10_list else 0.0,
        }
        metrics["per_class"] = per_class
        logging.info(
            "[%s] mAP=%.4f | R-Precision=%.4f nDCG@10=%.4f Hit@10=%.4f",
            label,
            metrics["mAP"],
            metrics["r_precision"],
            metrics["ndcg@10"],
            metrics["hit@10"],
        )
        return metrics

    def summary(self, args, args_dynamic, eval_summary: Dict):
        headers = [
            "query_mode",
            "model_name",
            "pretrained",
            "resume_post_train",
            "mAP",
            "r_precision",
            "ndcg@10",
            "hit@10",
        ]
        row = [
            args_dynamic.query_mode,
            args.model,
            args.pretrained,
            args.resume_post_train or "",
            round(eval_summary.get("mAP", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("r_precision", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("ndcg@10", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("hit@10", 0.0) * 100, 2) if eval_summary else "",
        ]
        return headers, row

    def save_metrics(self, output_dir: str, timestamp: str, eval_summary: Dict) -> None:
        per_class = eval_summary.get("per_class", {}) if eval_summary else {}
        if not per_class:
            return
        metrics_path = os.path.join(output_dir, f"{timestamp}.csv")
        with open(metrics_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "mAP", "recall@1", "recall@5", "recall@10", "precision@10"])
            for cls_name, metrics in sorted(per_class.items()):
                writer.writerow(
                    [
                        cls_name,
                        round(metrics.get("mAP", 0.0) * 100, 2),
                        round(metrics.get("recall@1", 0.0) * 100, 2),
                        round(metrics.get("recall@5", 0.0) * 100, 2),
                        round(metrics.get("recall@10", 0.0) * 100, 2),
                        round(metrics.get("precision@10", 0.0) * 100, 2),
                    ]
                )
        logging.info("Saved per-class metrics to %s", metrics_path)
