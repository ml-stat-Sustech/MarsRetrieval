import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

PROMPT_TEMPLATES = [
    "a photo of {}, a type of martian terrain",
    "a satellite photo of {}.",
    "a high-resolution remote sensing image of {} on Mars.",
]


class QueryImageDataset(Dataset):
    def __init__(self, image_paths: List[str], preprocess, return_path: bool = False):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.return_path = return_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if self.return_path:
            return path, path
        image = Image.open(path).convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)
        return image, path


def build_query_loader(
    image_paths: Iterable[str],
    preprocess,
    batch_size: int = 4,
    collate_fn=None,
    return_path: bool = False,
):
    dataset = QueryImageDataset(list(image_paths), preprocess, return_path=return_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def build_text_query(args, text_encoder, query_name: str, return_tensor: bool = False):
    if text_encoder is None:
        raise ValueError("Text encoder is required for text or multimodal queries.")

    target_phrase = query_name.replace("_", " ")
    text_inputs = [template.format(target_phrase) for template in PROMPT_TEMPLATES]

    with torch.no_grad():
        text_features = text_encoder.encode_text(text_inputs)
        if len(text_inputs) > 1:
            query = text_features.mean(dim=0, keepdim=True)
            query = F.normalize(query, p=2, dim=-1)
        else:
            query = text_features

    return query if return_tensor else query.cpu().numpy()


def build_image_query(args, image_encoder, reference_loader, text_features: Optional[torch.Tensor] = None):
    all_features = []
    for batch in tqdm(reference_loader, desc="Extracting query image features", unit="batch"):
        images, _ = batch
        image_features = image_encoder.encode_image(images)
        all_features.append(image_features)

    all_features = torch.cat(all_features, dim=0)

    if text_features is not None:
        text_features = text_features.to(all_features.device)
        similarities = all_features @ text_features.T
        weights = F.softmax(similarities / 0.1, dim=0)
        query_tensor = (weights * all_features).sum(dim=0, keepdim=True)
    else:
        query_tensor = torch.mean(all_features, dim=0, keepdim=True)

    query_tensor = F.normalize(query_tensor, p=2, dim=-1)
    query = query_tensor.cpu().numpy()

    if getattr(args, "feature_dim", None) is None:
        args.feature_dim = query.shape[1]
    elif query.shape[1] != args.feature_dim:
        raise ValueError(f"Image query dim mismatch: {query.shape[1]} != {args.feature_dim}")
    return query


def build_geolocalization_query(
    args,
    image_encoder,
    text_encoder,
    query_mode: str,
    query_images: Optional[Iterable[str]] = None,
    query_name: Optional[str] = None,
):
    logging.info("Building query in mode: %s", query_mode)

    def _expand_image_inputs(inputs: Iterable[str]) -> List[str]:
        expanded: List[str] = []
        for inp in inputs:
            p = Path(inp)
            if p.is_dir():
                for file in sorted(p.iterdir()):
                    if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                        expanded.append(str(file))
            else:
                expanded.append(str(p))
        return expanded

    reference_loader = None
    if query_images:
        image_list = _expand_image_inputs(query_images)
        if not image_list:
            raise ValueError("No images found for the provided query_images paths.")
        reference_loader = build_query_loader(
            image_list,
            image_encoder.get_processor(),
            batch_size=min(len(image_list), args.batch_size_database),
            collate_fn=getattr(image_encoder, "collate_fn", None),
            return_path=getattr(image_encoder, "use_path_inputs", False),
        )

    if query_mode == "image":
        if reference_loader is None:
            raise ValueError("query_images are required for image-based mode.")
        return build_image_query(args, image_encoder, reference_loader)

    if query_mode == "text":
        if query_name is None:
            raise ValueError("query_name is required for text-based mode.")
        return build_text_query(args, text_encoder, query_name)

    if query_mode == "hybrid":
        if reference_loader is None or query_name is None:
            raise ValueError("Hybrid-mode requires both query_images and query_name.")
        text_query_tensor = build_text_query(args, text_encoder, query_name, return_tensor=True)
        weighted_image_query = build_image_query(args, image_encoder, reference_loader, text_features=text_query_tensor)
        text_query_numpy = text_query_tensor.cpu().numpy()
        mixed_query = args.mix_image_ratio * weighted_image_query + (1 - args.mix_image_ratio) * text_query_numpy
        mixed_query /= np.linalg.norm(mixed_query, axis=1, keepdims=True)
        return mixed_query

    raise ValueError(f"Unsupported query_mode: {query_mode}")


#    # text_inputs = [
    #     "a Martian alluvial fans formed by fluvial sediment deposition",
    #     "a fan-shaped sedimentary landform on Mars formed by flowing water",
    #     "a radial alluvial fans on Mars with distributary channels",
    #     "an alluvial fans at the base of a Martian crater wall",
    #     "a subaerial alluvial fans on Mars without a steep frontal scarp"
    # ]

    # text_inputs = [
    #     "a Martian glacier-like form formed by ice-rich flow",
    #     "a lobate glacier-like landform on Mars",
    #     "a glacier-like form on Mars with flow-like surface textures",
    #     "a glacier-like form at the base of a Martian slope or crater wall",
    #     "a debris-covered glacier-like form on Mars"
    # ]

    # 
    # text_inputs = [
    #     "a Martian landslides formed by gravity-driven mass movement",
    #     "a large rock avalanche landslides on Mars",
    #     "a landslides deposit on Mars with chaotic hummocky terrain",
    #     "a landslides at the base of a Martian cliff or canyon wall",
    #     "a landslides deposit on Mars spreading outward from a steep slope"
    # ]

    # text_inputs = [
    #     "a Martian pitted cones with a central crater",
    #     "a circular cone-shaped landform on Mars with a prominent central pit",
    #     "a positive-relief pitted cones on Mars with smooth sloping sides",
    #     "a small conical landform on Mars featuring a summit crater",
    #     "a symmetric pitted cones landform on the Martian surface"
    # ]

    # text_inputs = [
    #     "a Martian yardang formed by wind-driven erosion",
    #     "an elongated yardang landform on Mars",
    #     "a yardang on Mars with streamlined ridge morphology",
    #     "a yardang field on Mars carved into bedrock by aeolian processes",
    #     "a yardang at the surface of Mars with parallel linear ridges"
    # ]

    # text_inputs = [
    # "a photo of pitted cones, a type of martian terrain",
    # "a satellite photo of pitted cones.",
    # "a high-resolution remote sensing image of pitted cones on Mars.",
    # ]