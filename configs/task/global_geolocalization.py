from dataclasses import dataclass
from ..config_base import Config

@dataclass
class GlobalGeoLocalization(Config):
    task_name = "global_geolocalization"
    project_dir = "data/global_geolocalization"
    logs = "./logs"
    ground_truth_csv = None
    seed = 1
    delta_degree = 0.2
    workers = 4
    batch_size_database = 256
    top_k = 20000
    eval_max_k = 20000
    radius_deg = 0.5
    feature_dim = None
    mix_image_ratio = 0.3
    thumb_resolution = 512

    def __post_init__(self):
        super().__post_init__()
        self.database_basedir = f"{self.project_dir}/database/image_size_{self.thumb_resolution}_delta_{self.delta_degree}"
        self.dataset_dir = f"{self.project_dir}/dataset"
        self.image_query_dir = f"{self.dataset_dir}/image_queries"
        self.thumb_dir = f"{self.dataset_dir}/tiles/image_size_{self.thumb_resolution}_delta_{self.delta_degree}/thumb"
        self.global_img_dir = f"{self.dataset_dir}/mars_global.jpg"