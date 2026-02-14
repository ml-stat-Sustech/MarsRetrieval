from .geolocalization_dataset import GeoLocalizationDatabaseBuilder
from .landform_dataset import LandformRetrievalDatabaseBuilder
from .paired_vl_dataset import PairedVLRetrievalDatabaseBuilder


def build_dataset(args, image_encoder, text_encoder=None, delta: float = 0.2):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        builder = GeoLocalizationDatabaseBuilder()
        return builder.build(args, image_encoder, delta=delta)
    if task_name == "landform_retrieval":
        builder = LandformRetrievalDatabaseBuilder()
        return builder.build(args, image_encoder, delta=delta)
    if task_name == "paired_image_text_retrieval":
        builder = PairedVLRetrievalDatabaseBuilder()
        return builder.build(args, image_encoder, text_encoder, delta=delta)
    raise ValueError(f"Unsupported task_name for dataset build: {task_name}")


def build_dataset_distributed(args, image_encoder, delta: float, rank: int, world_size: int):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        builder = GeoLocalizationDatabaseBuilder()
        return builder.build_distributed(args, image_encoder, delta=delta, rank=rank, world_size=world_size)
    if task_name == "landform_retrieval":
        builder = LandformRetrievalDatabaseBuilder()
        return builder.build_distributed(args, image_encoder, delta=delta, rank=rank, world_size=world_size)
    if task_name == "paired_image_text_retrieval":
        raise NotImplementedError("Distributed DB build not implemented for cross-modal matching.")
    raise ValueError(f"Unsupported task_name for dataset build: {task_name}")
