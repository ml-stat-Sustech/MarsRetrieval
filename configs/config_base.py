import os
import importlib
from dataclasses import dataclass


@dataclass
class Config:
    # --------------------- General ---------------------
    name = "open-clip"
    logs = "./logs/"
    project_name = None
    task_name = None
    precision = "amp"
    seed = 0

    # --------------------- Encoders/Model ---------------------
    image_encoder_type = "openclip"
    text_encoder_type = None
    model = "RN50"
    pretrained = ""
    force_quick_gelu = False
    torchscript = False
    force_custom_text = None
    force_patch_dropout = None
    force_image_size = None
    image_mean = None
    image_std = None
    siglip = False
    resume_post_train = None

    # --------------------- Task-Specific (override in task configs) ---------------------
    project_dir = None
    database_basedir = None
    delta_degree = None
    workers = None
    batch_size_database = None
    top_k = None
    eval_max_k = None
    radius_deg = None
    feature_dim = None
    mix_image_ratio = None
    dataset_dir = None

    def __post_init__(self):
        args = self
        args.name = self.__class__.__name__
        args.output_dir = os.path.join(args.logs, args.name)


def _iter_model_override_attrs(cfg):
    for name, value in cfg.__class__.__dict__.items():
        if name.startswith("_"):
            continue
        if callable(value):
            continue
        yield name


def _load_config_from_package(config_name: str, package: str, label: str):
    base_dir = os.path.dirname(__file__)
    package_dir = os.path.join(base_dir, package)
    all_configs = {}
    for file_name in os.listdir(package_dir):
        if file_name.endswith(".py") and not file_name.startswith("__"):
            module_name = file_name[:-3]
            full_module_name = f"{__package__}.{package}.{module_name}"
            module = importlib.import_module(full_module_name)
            for attr_name in dir(module):
                if attr_name in ["Config"] or attr_name.startswith("__"):
                    continue
                obj = getattr(module, attr_name)
                if isinstance(obj, type) and attr_name not in all_configs:
                    all_configs[attr_name] = module

    if config_name not in all_configs:
        raise KeyError(f"Config {config_name} not found in {label} configs.")
    return getattr(all_configs[config_name], config_name)()


def load_task_config(config_name: str):
    return _load_config_from_package(config_name, package="task", label="task")


def load_model_config(config_name: str):
    return _load_config_from_package(config_name, package="model", label="model")


def merge_configs(task_cfg, model_cfg):
    if model_cfg is None:
        return task_cfg
    for name in _iter_model_override_attrs(model_cfg):
        value = getattr(model_cfg, name)
        if value is not None:
            setattr(task_cfg, name, value)
    return task_cfg
