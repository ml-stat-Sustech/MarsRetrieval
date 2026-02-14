from dataclasses import dataclass
from ..config_base import Config



@dataclass
class LandformRetrieval(Config):
    task_name = "landform_retrieval"
    project_dir = "data/landform_retrieval"
    logs = "./logs"
    seed = 1
    workers = 4
    batch_size_database = 256
    top_k = None
    eval_max_k = None
    feature_dim = None

    def __post_init__(self):
        super().__post_init__()
        self.database_basedir = f"{self.project_dir}/database"
        self.dataset_dir = f"{self.project_dir}/dataset"
