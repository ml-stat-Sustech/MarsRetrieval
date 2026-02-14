from dataclasses import dataclass
from ..config_base import Config


@dataclass
class OpenCLIP(Config):
    image_encoder_type = "openclip"
    text_encoder_type = "openclip"


@dataclass
class CLIPMarScope(Config):
    image_encoder_type = "openclip"
    text_encoder_type = "openclip"
    model = "ViT-L-14-quickgelu"
    pretrained = "dfn2b"
    force_image_size = 512
    force_quick_gelu = True
    batch_size_database = 128


@dataclass
class E5V(Config):
    image_encoder_type = "e5-v"
    text_encoder_type = "e5-v"
    model = "royokong/e5-v"
    batch_size_database = 1
    patch_size = 14


@dataclass
class VLM2Vec(Config):
    image_encoder_type = "vlm2vec"
    text_encoder_type = "vlm2vec"
    model = "VLM2Vec/VLM2Vec-V2.0"
    batch_size_database = 8


@dataclass
class AimV2VL(Config):
    image_encoder_type = "aimv2_vl"
    text_encoder_type = "aimv2_vl"
    model = "apple/aimv2-large-patch14-224-lit"
    pretrained = "c2cd59a786c4c06f39d199c50d08cc2eab9f8605"
    batch_size_database = 96


@dataclass
class AimV2Vis(Config):
    image_encoder_type = "aimv2_vis"
    text_encoder_type = "none"
    model = "apple/aimv2-large-patch14-448"
    pretrained = "cefb13f21003bdadba65bfbee956c82b976cd23d"
    batch_size_database = 64


@dataclass
class B3Qwen2(Config):
    image_encoder_type = "vlm2vec"
    text_encoder_type = "vlm2vec"
    model = "raghavlite/B3_Qwen2_2B"
    batch_size_database = 8


@dataclass
class OpsMM(Config):
    image_encoder_type = "opsmm_v1"
    text_encoder_type = "opsmm_v1"
    model = "OpenSearch-AI/Ops-MM-embedding-v1-2B"
    batch_size_database = 8


@dataclass
class Qwen3VLEmbedding(Config):
    image_encoder_type = "qwen3_vl_embedding"
    text_encoder_type = "qwen3_vl_embedding"
    model = "Qwen/Qwen3-VL-Embedding-2B"
    batch_size_database = 16


@dataclass
class GME(Config):
    image_encoder_type = "gme"
    text_encoder_type = "gme"
    model = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    batch_size_database = 4
    gme_text_instruction = "Find an image that matches the given text."


@dataclass
class DinoV3(Config):
    image_encoder_type = "dinov3"
    text_encoder_type = "none"
    model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    pretrained = "hf"


@dataclass
class BGEVL(Config):
    image_encoder_type = "bge-vl"
    text_encoder_type = "bge-vl"
    model = "BAAI/BGE-VL-large"
    pretrained = "bge-vl"


@dataclass
class Jina(Config):
    image_encoder_type = "jina"
    text_encoder_type = "jina"
    model = "jinaai/jina-embeddings-v4"
    pretrained = "hf"
    batch_size_database = 4
