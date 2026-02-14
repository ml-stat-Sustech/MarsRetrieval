from dataclasses import dataclass
from ..config_base import Config


@dataclass
class PairedVLRetrieval(Config):
    task_name = "paired_image_text_retrieval"
    project_dir = "data/paired_image_text_retrieval"
    caption_key = "refined_caption" # ori_caption refined_caption
    logs = "./logs"
    seed = 1
    workers = 4
    batch_size_database = 64
    top_k = 10
    feature_dim = None

    def __post_init__(self):
        super().__post_init__()
        self.database_basedir = f"{self.project_dir}/database"
        self.dataset_dir = f"{self.project_dir}/dataset"

