import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from .base import EvaluatorBase

MAP_WIDTH = 2644
MAP_HEIGHT = 1322
SHIFT_X = 158
SHIFT_Y = 68


def lonlat_to_pixel(
    lon,
    lat,
    map_width: int = MAP_WIDTH,
    map_height: int = MAP_HEIGHT,
    shift_x: int = SHIFT_X,
    shift_y: int = SHIFT_Y,
):
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    px = (lon + 180) / 360 * map_width + shift_x
    py = (90 - lat) / 180 * map_height + shift_y
    return px, py


def load_csv(gt_csv_path: str):
    df = pd.read_csv(gt_csv_path)
    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("GT CSV must contain lon, lat columns.")
    return df


def eval_points_coverage(gt_df, pred_df, radius_px=None, radius_deg=None):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scipy is required for evaluation.") from exc

    if radius_deg is not None:
        radius_px = radius_deg * (MAP_WIDTH / 360.0)

    if radius_px is None:
        raise ValueError("radius_px and radius_deg cannot both be None")

    gt_px, gt_py = lonlat_to_pixel(gt_df["lon"].values, gt_df["lat"].values)
    pred_px, pred_py = lonlat_to_pixel(pred_df["lon"].values, pred_df["lat"].values)

    gt_pts = np.stack([gt_px, gt_py], axis=1)
    pred_pts = np.stack([pred_px, pred_py], axis=1)

    n_gt = len(gt_pts)
    n_pred = len(pred_pts)
    if n_gt == 0 or n_pred == 0:
        return {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "gt_hit": 0,
            "pred_hit": 0,
            "gt_dist_mean": None,
            "pred_dist_mean": None,
            "radius_px": radius_px,
            "radius_deg": radius_deg,
        }

    tree_pred = cKDTree(pred_pts)
    d_gt, _ = tree_pred.query(gt_pts, k=1)
    gt_hit_mask = d_gt <= radius_px
    gt_hit = int(gt_hit_mask.sum())

    tree_gt = cKDTree(gt_pts)
    d_pred, _ = tree_gt.query(pred_pts, k=1)
    pred_hit_mask = d_pred <= radius_px
    pred_hit = int(pred_hit_mask.sum())

    recall = gt_hit / n_gt if n_gt > 0 else 0.0
    precision = pred_hit / n_pred if n_pred > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "gt_hit": gt_hit,
        "pred_hit": pred_hit,
        "gt_dist_mean": float(d_gt[gt_hit_mask].mean()) if gt_hit > 0 else None,
        "pred_dist_mean": float(d_pred[pred_hit_mask].mean()) if pred_hit > 0 else None,
        "radius_px": radius_px,
        "radius_deg": radius_deg,
    }


class GeoLocalizationEvaluator(EvaluatorBase):
    def __init__(self, gt_csv_path: Optional[str], radius_deg: float = 0.5, max_k: int = 200000, points: int = 100):
        self.radius_deg = radius_deg
        self.max_k = max_k
        self.points = points
        self.df_gt = load_csv(gt_csv_path) if gt_csv_path else None

    def evaluate(self, pred_df: pd.DataFrame, label: str = "run") -> Dict:
        if self.df_gt is None:
            logging.info("No ground-truth CSV provided. Skipping evaluation.")
            return {}

        df_sorted = pred_df.sort_values("similarity", ascending=False).reset_index(drop=True)
        n_pred = len(df_sorted)
        if n_pred == 0:
            logging.warning("Prediction dataframe is empty, skipping evaluation.")
            return {"best": None, "auprc": 0.0, "curve": []}

        max_k = min(n_pred, self.max_k)
        step = max(1, max_k // self.points)
        k_values = list(range(step, max_k + 1, step))
        if k_values[-1] != max_k:
            k_values.append(max_k)

        best = None
        curve = []
        precisions = []
        recalls = []
        for k in k_values:
            metrics = eval_points_coverage(self.df_gt, df_sorted.head(k), radius_deg=self.radius_deg)
            metrics["k"] = k
            curve.append(metrics)
            if best is None or metrics["f1"] > best["f1"]:
                best = {"k": k, "f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"]}
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])

        if recalls:
            r_array = np.array(recalls)
            p_array = np.array(precisions)
            order = np.argsort(r_array)
            sorted_r = r_array[order]
            sorted_p = p_array[order]
            auprc = auc(sorted_r, sorted_p)
        else:
            auprc = 0.0
            sorted_r, sorted_p = np.array([]), np.array([])

        logging.info(
            "[%s] Best F1=%.4f at K=%s (precision=%.4f, recall=%.4f) | AUPRC=%.4f",
            label,
            best["f1"],
            best["k"],
            best["precision"],
            best["recall"],
            auprc,
        )

        return {"best": best, "curve": curve, "auprc": auprc}

    def summary(self, args, args_dynamic, eval_summary: Dict):
        best_f1 = None
        auprc_score = None
        if eval_summary:
            best = eval_summary.get("best") or {}
            best_f1 = (
                round(best.get("f1", 0.0) * 100, 2),
                round(best.get("precision", 0.0) * 100, 2),
                round(best.get("recall", 0.0) * 100, 2),
                best.get("k"),
            )
            auprc_score = round(eval_summary.get("auprc", 0.0) * 100, 2) if "auprc" in eval_summary else None

        headers = [
            "query_text",
            "query_mode",
            "model_name",
            "pretrained",
            "resume_post_train",
            "f1",
            "precision",
            "recall",
            "auprc",
            "k_at_best",
        ]
        row = [
            args_dynamic.query_text or "",
            args_dynamic.query_mode,
            args.model,
            args.pretrained,
            args.resume_post_train or "",
            best_f1[0] if best_f1 else "",
            best_f1[1] if best_f1 else "",
            best_f1[2] if best_f1 else "",
            auprc_score if auprc_score is not None else "",
            best_f1[3] if best_f1 else "",
        ]
        return headers, row
