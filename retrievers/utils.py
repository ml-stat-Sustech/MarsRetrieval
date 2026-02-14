from .geolocalization_retriever import GeoLocalizationRetriever
from .landform_retriever import LandformRetriever
from .paired_vl_retriever import PairedVLRetriever


def build_retriever(args, database):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        return GeoLocalizationRetriever(args=args, database=database)
    if task_name == "landform_retrieval":
        return LandformRetriever(args=args, database=database)
    if task_name == "paired_image_text_retrieval":
        return PairedVLRetriever(args=args, database=database)
    raise ValueError(f"Unsupported task_name for retriever build: {task_name}")
