from .geolocalization import build_geolocalization_query
from .landform_retrieval import build_landform_query


def build_query(
    args,
    image_encoder,
    text_encoder,
    query_mode: str,
    query_images=None,
    query_name=None,
):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        return build_geolocalization_query(
            args,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            query_mode=query_mode,
            query_images=query_images,
            query_name=query_name,
        )
    if task_name == "landform_retrieval":
        return build_landform_query(
            args,
            text_encoder=text_encoder,
            query_mode=query_mode,
        )
    if task_name == "paired_image_text_retrieval":
        raise ValueError("Cross-modal matching does not use build_query; compute retrieval directly.")
    raise ValueError(f"Unsupported task_name for queries: {task_name}")
