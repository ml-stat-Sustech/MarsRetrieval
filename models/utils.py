import logging
from typing import Optional

from .image_encoder.base import ImageEncoderBase
from .image_encoder.bgevl_image_encoder import BGEVLImageEncoder
from .image_encoder.dinov3_image_encoder import DinoV3ImageEncoder
from .image_encoder.jina_image_encoder import JinaImageEncoder
from .image_encoder.e5v_image_encoder import E5VImageEncoder
from .image_encoder.aimv2_vl_image_encoder import AimV2VLImageEncoder
from .image_encoder.aimv2_vis_image_encoder import AimV2VisImageEncoder
from .image_encoder.openclip_image_encoder import OpenCLIPImageEncoder
from .image_encoder.gme_image_encoder import GMEImageEncoder
from .bgevl import BGEVLComponents, build_bgevl_components
from .e5v import E5VComponents, build_e5v_components
from .aimv2_vl import AimV2VLComponents, build_aimv2_vl_components
from .aimv2_vis import AimV2VisComponents, build_aimv2_vis_components
from .jina import JinaComponents, build_jina_components
from .openclip import OpenCLIPComponents, build_openclip_components
from .gme import GMEComponents, build_gme_components
from .text_encoder.base import TextEncoderBase
from .text_encoder.bgevl_text_encoder import BGEVLTextEncoder
from .text_encoder.e5v_text_encoder import E5VTextEncoder
from .text_encoder.aimv2_vl_text_encoder import AimV2VLTextEncoder
from .text_encoder.jina_text_encoder import JinaTextEncoder
from .text_encoder.openclip_text_encoder import OpenCLIPTextEncoder
from .text_encoder.gme_text_encoder import GMETextEncoder


def _infer_image_encoder_type(args) -> str:
    if getattr(args, "image_encoder_type", None):
        return args.image_encoder_type
    model = getattr(args, "model", "")
    if "/" in model:
        family, _ = model.split("/", 1)
        family = family.lower()
        if family == "jinaai":
            return "jina"
        if "bge-vl" in model.lower():
            return "bge-vl"
        if "e5-v" in model.lower():
            return "e5-v"
        if "vlm2vec" in model.lower():
            return "vlm2vec"
        if "b3" in model.lower():
            return "vlm2vec"
        if "ops-mm" in model.lower() or "ops_mm" in model.lower() or "opensearch-ai" in model.lower():
            return "opsmm_v1"
        if "qwen3-vl-embedding" in model.lower() or "qwen3_vl_embedding" in model.lower():
            return "qwen3_vl_embedding"
        if "aimv" in model.lower() or family == "apple":
            if "patch14-448" in model.lower():
                return "aimv2_vis"
            return "aimv2_vl"
        if "gme" in model.lower() or "qwen2-vl" in model.lower():
            return "gme"
        return family
    return "openclip"


def _get_openclip_components(args, device) -> OpenCLIPComponents:
    if getattr(args, "_openclip_components", None) is None:
        args._openclip_components = build_openclip_components(args, device)
        args.preprocess_train = args._openclip_components.preprocess_train
        args.preprocess_val = args._openclip_components.preprocess_val
    return args._openclip_components


def _get_jina_components(args, device) -> JinaComponents:
    if getattr(args, "_jina_components", None) is None:
        args._jina_components = build_jina_components(args, device)
    return args._jina_components


def _get_gme_components(args, device) -> GMEComponents:
    if getattr(args, "_gme_components", None) is None:
        args._gme_components = build_gme_components(args, device)
    return args._gme_components


def _get_bgevl_components(args, device) -> BGEVLComponents:
    if getattr(args, "_bgevl_components", None) is None:
        args._bgevl_components = build_bgevl_components(args, device)
    return args._bgevl_components


def _get_e5v_components(args, device) -> E5VComponents:
    if getattr(args, "_e5v_components", None) is None:
        args._e5v_components = build_e5v_components(args, device)
    return args._e5v_components


def _get_vlm2vec_components(args, device):
    if getattr(args, "_vlm2vec_components", None) is None:
        from .vlm2vec import build_vlm2vec_components

        args._vlm2vec_components = build_vlm2vec_components(args, device)
    return args._vlm2vec_components


def _get_aimv2_vl_components(args, device) -> AimV2VLComponents:
    if getattr(args, "_aimv2_vl_components", None) is None:
        args._aimv2_vl_components = build_aimv2_vl_components(args, device)
    return args._aimv2_vl_components


def _get_aimv2_vis_components(args, device) -> AimV2VisComponents:
    if getattr(args, "_aimv2_vis_components", None) is None:
        args._aimv2_vis_components = build_aimv2_vis_components(args, device)
    return args._aimv2_vis_components


def _get_opsmm_v1_components(args, device):
    if getattr(args, "_opsmm_v1_components", None) is None:
        from .opsmm_v1 import build_opsmm_v1_components

        args._opsmm_v1_components = build_opsmm_v1_components(args, device)
    return args._opsmm_v1_components


def _get_qwen3_vl_embedding_components(args, device):
    if getattr(args, "_qwen3_vl_embedding_components", None) is None:
        from .qwen3_vl_embedding import build_qwen3_vl_embedding_components

        args._qwen3_vl_embedding_components = build_qwen3_vl_embedding_components(args, device)
    return args._qwen3_vl_embedding_components


def build_image_encoder(args, device) -> ImageEncoderBase:
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name not in ("global_geolocalization", "landform_retrieval", "paired_image_text_retrieval"):
        raise ValueError(f"Unsupported task_name for image encoder: {task_name}")
    encoder_type = _infer_image_encoder_type(args)
    if encoder_type == "openclip":
        components = _get_openclip_components(args, device)
        return OpenCLIPImageEncoder(components, device)

    if encoder_type.lower() == "dinov3":
        model_id = getattr(args, "model", None) or "facebook/dinov3-vitl16-pretrain-lvd1689m"
        pooling = getattr(args, "dinov3_pooling", "cls")
        return DinoV3ImageEncoder(model_id, device, pooling=pooling)

    if encoder_type == "jina":
        components = _get_jina_components(args, device)
        return JinaImageEncoder(components)

    if encoder_type == "bge-vl":
        components = _get_bgevl_components(args, device)
        return BGEVLImageEncoder(components)

    if encoder_type == "e5-v":
        components = _get_e5v_components(args, device)
        return E5VImageEncoder(components)

    if encoder_type == "vlm2vec":
        components = _get_vlm2vec_components(args, device)
        from .image_encoder.vlm2vec_image_encoder import VLM2VecImageEncoder

        return VLM2VecImageEncoder(components)

    if encoder_type == "aimv2_vl":
        components = _get_aimv2_vl_components(args, device)
        return AimV2VLImageEncoder(components)

    if encoder_type == "aimv2_vis":
        components = _get_aimv2_vis_components(args, device)
        return AimV2VisImageEncoder(components)

    if encoder_type == "opsmm_v1":
        components = _get_opsmm_v1_components(args, device)
        from .image_encoder.opsmm_v1_image_encoder import OpsMMV1ImageEncoder

        return OpsMMV1ImageEncoder(components)

    if encoder_type == "qwen3_vl_embedding":
        components = _get_qwen3_vl_embedding_components(args, device)
        from .image_encoder.qwen3_vl_embedding_image_encoder import Qwen3VLEmbeddingImageEncoder

        return Qwen3VLEmbeddingImageEncoder(components)

    if encoder_type == "gme":
        components = _get_gme_components(args, device)
        return GMEImageEncoder(components)

    raise ValueError(f"Unsupported image encoder type: {encoder_type}")


def build_text_encoder(args, device) -> Optional[TextEncoderBase]:
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name not in ("global_geolocalization", "landform_retrieval", "paired_image_text_retrieval"):
        raise ValueError(f"Unsupported task_name for text encoder: {task_name}")
    encoder_type = getattr(args, "text_encoder_type", None) or "openclip"
    if encoder_type == "openclip":
        components = _get_openclip_components(args, device)
        return OpenCLIPTextEncoder(components, device)

    if encoder_type == "jina":
        components = _get_jina_components(args, device)
        return JinaTextEncoder(components)

    if encoder_type == "bge-vl":
        components = _get_bgevl_components(args, device)
        return BGEVLTextEncoder(components)

    if encoder_type == "e5-v":
        components = _get_e5v_components(args, device)
        return E5VTextEncoder(components)

    if encoder_type == "vlm2vec":
        components = _get_vlm2vec_components(args, device)
        from .text_encoder.vlm2vec_text_encoder import VLM2VecTextEncoder

        return VLM2VecTextEncoder(components)

    if encoder_type == "aimv2_vl":
        components = _get_aimv2_vl_components(args, device)
        return AimV2VLTextEncoder(components)

    if encoder_type == "opsmm_v1":
        components = _get_opsmm_v1_components(args, device)
        from .text_encoder.opsmm_v1_text_encoder import OpsMMV1TextEncoder

        return OpsMMV1TextEncoder(components)

    if encoder_type == "qwen3_vl_embedding":
        components = _get_qwen3_vl_embedding_components(args, device)
        from .text_encoder.qwen3_vl_embedding_text_encoder import Qwen3VLEmbeddingTextEncoder

        return Qwen3VLEmbeddingTextEncoder(components)

    if encoder_type == "aimv2_vis":
        raise ValueError("Text encoder is not available for aimv2-vis (vision-only) models.")

    if encoder_type == "gme":
        components = _get_gme_components(args, device)
        return GMETextEncoder(components)

    if encoder_type == "none":
        logging.info("Text encoder disabled (type=none).")
        return None

    raise ValueError(f"Unsupported text encoder type: {encoder_type}")
