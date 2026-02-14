from .geolocalization_evaluator import GeoLocalizationEvaluator
from .landform_evaluator import LandformlEvaluator
from .paired_vl_evaluator import PairedVLEvaluator


def build_evaluator(args):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        gt_csv_path = getattr(args, "ground_truth_csv", None)
        if gt_csv_path is None:
            return None
        return GeoLocalizationEvaluator(
            gt_csv_path=gt_csv_path,
            radius_deg=getattr(args, "radius_deg", None),
            max_k=getattr(args, "eval_max_k", None),
        )
    if task_name == "landform_retrieval":
        return LandformlEvaluator(
            gt_map=getattr(args, "landform_gt", None),
            max_k=getattr(args, "eval_max_k", None),
        )
    if task_name == "paired_image_text_retrieval":
        return PairedVLEvaluator()
    raise ValueError(f"Unsupported task_name for evaluator build: {task_name}")
