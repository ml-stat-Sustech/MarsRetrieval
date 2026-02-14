import argparse
import datetime
import logging
import os
import sys

import torch
import torch.distributed as dist

from configs.config_base import load_task_config, load_model_config, merge_configs
from tools.utils import _merge_args, _configure_logging, random_seed, _silence_noisy_loggers


def _parse_args():
    parser = argparse.ArgumentParser(description="Distributed database builder")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name.")
    parser.add_argument("--model_config", type=str, required=True, help="Model config name.")
    parser.add_argument("--task_name", type=str, default=None, help="Task name for module dispatch.")
    parser.add_argument("--model_name", type=str, default=None, help="Model name override.")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained tag override.")
    parser.add_argument("--resume_post_train", type=str, default=None, help="Checkpoint for pretrained weights.")
    parser.add_argument("--image_encoder_type", type=str, default=None, help="Image encoder type.")
    parser.add_argument("--text_encoder_type", type=str, default="none", help="Text encoder type (unused for DB build).")
    parser.add_argument("--query_mode", type=str, default="image", help="Query mode placeholder.")
    parser.add_argument("--query_images", nargs="*", default=None, help="Unused placeholder.")
    parser.add_argument("--query_text", type=str, default=None, help="Unused placeholder.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (for logs).")
    parser.add_argument("--output_dir", type=str, default=None, help="Unused here.")
    parser.add_argument("--delta_degree", type=float, default=None, help="Delta degree for DB build.")
    parser.add_argument("--dinov3_pooling", type=str, default=None, help="Pooling for DINOv3 (cls|mean).")
    parser.add_argument("--batch_size_database", type=int, default=None, help="Batch size for DB build.")
    parser.add_argument("--top_k", type=int, default=None, help="Unused placeholder.")
    parser.add_argument("--db_dir", type=str, default=None, help="Optional database directory override.")
    parser.add_argument("--radius_deg", type=float, default=None, help="Unused placeholder.")
    parser.add_argument("--eval_max_k", type=int, default=None, help="Unused placeholder.")
    parser.add_argument("--ground_truth_csv", type=str, default=None, help="Unused placeholder.")
    return parser.parse_args()


def init_distributed():
    # torchrun should set these env vars; if missing, the launch method is wrong.
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    timeout_seconds = int(os.getenv("TORCH_DISTRIBUTED_TIMEOUT", "86400"))
    timeout = datetime.timedelta(seconds=timeout_seconds)

    # Critical: don't let an "early init" happen silently.
    if dist.is_initialized():
        raise RuntimeError(
            f"Process group already initialized BEFORE init_distributed(). "
            f"Your timeout/backend won't take effect. "
            f"(rank={rank}, local_rank={local_rank})"
        )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timeout,
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    print(
        f"[rank {rank}] init ok: backend={dist.get_backend()} "
        f"cuda={cuda_ok} local_rank={local_rank} timeout_s={timeout_seconds}",
        flush=True
    )
    return rank, world_size, local_rank



def main():
    rank, world_size, local_rank = init_distributed()

    if rank == 0:
        logging.info("Distributed initialized. Backend: %s", dist.get_backend())

    from mars_datasets.utils import build_dataset_distributed
    from models.utils import build_image_encoder

    args_dyn = _parse_args()
    task_cfg = load_task_config(args_dyn.task_config)
    model_cfg = load_model_config(args_dyn.model_config)
    args = merge_configs(task_cfg, model_cfg)
    args = _merge_args(args, args_dyn)

    _silence_noisy_loggers()

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None

    if rank == 0:
        log_dir = os.path.join(args.logs, args.name)
        _configure_logging(log_dir)
        logging.info("World size: %s", world_size)
        logging.info("Using device: %s", device)
        
    dist.barrier(device_ids=[local_rank])

    random_seed(args.seed + rank)

    image_encoder = build_image_encoder(args, device)
    if args_dyn.delta_degree is not None:
        delta = args_dyn.delta_degree
    elif getattr(args, "delta_degree", None) is not None:
        delta = args.delta_degree
    else:
        delta = 0.2

    result = build_dataset_distributed(args, image_encoder, delta=delta, rank=rank, world_size=world_size)
    if rank == 0:
        logging.info("DB build complete. Saved to %s", result.get("db_dir", ""))

    logging.info(f"Rank {rank} finished, waiting for others...")
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
