import logging
import os
import pickle
import re
import tarfile
from typing import Dict, Tuple, List, Optional

import faiss
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset, IterableDataset, get_worker_info
from tqdm import tqdm
import torch.distributed as dist
from numpy.lib.format import open_memmap

from .base import DatasetBuilderBase

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _extract_coordinates(image_name: str) -> Tuple[float, float]:
    image_name = os.path.basename(image_name)
    match = re.match(r"^(E_|E)(\d+_\d+)_(N|N_)(\d+_\d+)\.png$", image_name)
    if match:
        lon_dir, lon_val_str, lat_dir, lat_val_str = match.groups()
        lon_sign = -1 if lon_dir == "E_" else 1
        lat_sign = 1 if lat_dir == "N" else -1
        lon_val = float(lon_val_str.replace("_", "."))
        lat_val = float(lat_val_str.replace("_", "."))
        return lon_sign * lon_val, lat_sign * lat_val
    return None, None


class MarsBenchmarkDataset(Dataset):
    def __init__(self, thumb_dir: str, transform=None, return_path: bool = False):
        self.thumb_dir = thumb_dir
        self.transform = transform
        self.return_path = return_path
        self.samples_pkl_path = os.path.join(os.path.dirname(thumb_dir), "samples.pkl")
        self.samples = self._load_or_scan_samples()

    def _load_or_scan_samples(self):
        if os.path.exists(self.samples_pkl_path):
            logging.info("Loading samples from %s", self.samples_pkl_path)
            with open(self.samples_pkl_path, "rb") as f:
                return pickle.load(f)

        logging.info("No samples.pkl found, scanning %s ...", self.thumb_dir)
        samples = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        for root, dirs, files in os.walk(self.thumb_dir):
            dirs.sort()
            for fname in sorted(files):
                if not fname.lower().endswith(valid_extensions):
                    continue
                rel_path = os.path.relpath(os.path.join(root, fname), self.thumb_dir)
                samples.append(rel_path)

        with open(self.samples_pkl_path, "wb") as f:
            pickle.dump(samples, f)
        logging.info("Saved %d samples to %s", len(samples), self.samples_pkl_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name = self.samples[index]
        path = os.path.join(self.thumb_dir, image_name)
        if self.return_path:
            return path, image_name
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name


class MarsBenchmarkWDSDataset(IterableDataset):
    def __init__(
        self,
        tar_paths: List[str],
        transform=None,
        show_tar_progress: bool = False,
        progress_desc: str = None,
        progress_position: int = 0,
    ):
        self.tar_paths = tar_paths
        self.transform = transform
        self.show_tar_progress = show_tar_progress
        self.progress_desc = progress_desc
        self.progress_position = progress_position

    def _iter_tar(self, tar_path: str):
        with tarfile.open(tar_path) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    image = Image.open(f).convert("RGB")
                except Exception:
                    continue
                if self.transform is not None:
                    image = self.transform(image)
                image_name = os.path.basename(name)
                yield image, image_name

    def __iter__(self):
        worker_info = get_worker_info()
        tar_paths = self.tar_paths
        if worker_info is not None:
            per_worker = int(np.ceil(len(tar_paths) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(tar_paths))
            tar_paths = tar_paths[start:end]
        if self.show_tar_progress and worker_info is None:
            tar_iter = tqdm(
                tar_paths,
                total=len(tar_paths),
                desc=self.progress_desc or "Processing tar files",
                unit="tar",
                position=self.progress_position,
                leave=False,
                dynamic_ncols=True,
            )
        else:
            tar_iter = tar_paths
        for tar_path in tar_iter:
            yield from self._iter_tar(tar_path)

class GeoLocalizationDatabaseBuilder(DatasetBuilderBase):
    def _resolve_paths(self, args, delta: float) -> Tuple[str, str]:
        if getattr(args, "db_dir", None):
            db_dir = args.db_dir
        elif getattr(args, "resume_post_train", None):
            tag = args.resume_post_train.strip("/").split("/")[-3]
            base_dir = getattr(args, "database_basedir", None) or getattr(args, "project_dir", ".")
            db_dir = f"{base_dir}/{tag}"
        else:
            raise ValueError("db_dir or resume_post_train must be provided to locate the database.")

        thumb_dir = getattr(args, "thumb_dir", None)
        if not thumb_dir:
            dataset_dir = getattr(args, "dataset_dir", None) or f"{args.project_dir}/dataset"
            thumb_dir = f"{dataset_dir}/tiles/thumb"
        return db_dir, thumb_dir

    def _find_tar_paths(self, thumb_dir: str) -> List[str]:
        if not os.path.isdir(thumb_dir):
            return []
        tar_paths = []
        for root, _, files in os.walk(thumb_dir):
            for fname in files:
                if fname.endswith(".tar"):
                    tar_paths.append(os.path.join(root, fname))
        return sorted(tar_paths)

    def build(self, args, image_encoder, delta: float) -> Dict:
        db_dir, thumb_dir = self._resolve_paths(args, delta)
        logging.info("Building database at delta=%s -> %s", delta, db_dir)

        os.makedirs(db_dir, exist_ok=True)

        feature_save_path = os.path.join(db_dir, "features.npy")
        metadata_save_path = os.path.join(db_dir, "metadata.pkl")
        coordinates_save_path = os.path.join(db_dir, "coordinates.pkl")

        if (
            os.path.exists(feature_save_path)
            and os.path.exists(metadata_save_path)
            and os.path.exists(coordinates_save_path)
        ):
            logging.info("Loading cached database from %s", db_dir)
            features = np.load(feature_save_path)
            with open(metadata_save_path, "rb") as f:
                metadata = pickle.load(f)
            with open(coordinates_save_path, "rb") as f:
                coordinates = pickle.load(f)
            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = features.shape[1]
        else:
            logging.info("Preparing database features from thumbnails in %s", thumb_dir)
            tar_paths = self._find_tar_paths(thumb_dir)
            use_path_inputs = getattr(image_encoder, "use_path_inputs", False)
            if tar_paths:
                if use_path_inputs:
                    raise ValueError("WDS thumbnails do not support path-based encoders.")
                show_tar_progress = os.getenv("WDS_TAR_PROGRESS", "1").strip() == "1"
                loader_workers = 0 if show_tar_progress else args.workers
                if show_tar_progress and args.workers > 0:
                    logging.info("WDS tar progress enabled; forcing DataLoader workers=0 for visible tqdm.")
                target_dataset = MarsBenchmarkWDSDataset(
                    tar_paths,
                    transform=image_encoder.get_processor(),
                    show_tar_progress=show_tar_progress,
                    progress_desc="Building database (tar)",
                )
            else:
                target_dataset = MarsBenchmarkDataset(
                    thumb_dir,
                    transform=image_encoder.get_processor(),
                    return_path=use_path_inputs,
                )
                loader_workers = args.workers
            target_loader = DataLoader(
                target_dataset,
                batch_size=args.batch_size_database,
                shuffle=False,
                num_workers=loader_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=getattr(image_encoder, "collate_fn", None),
            )

            features = []
            metadata = []
            coordinates = []

            for images, image_names in tqdm(target_loader, desc="Building database", unit="batch"):
                image_features = image_encoder.encode_image(images)
                image_features = image_features.cpu().numpy()

                features.append(image_features)
                metadata.extend(image_names)
                batch_coordinates = [_extract_coordinates(name) for name in image_names]
                coordinates.extend(batch_coordinates)

            features = np.concatenate(features, axis=0).astype("float32")

            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = features.shape[1]

            np.save(feature_save_path, features)
            with open(metadata_save_path, "wb") as f:
                pickle.dump(metadata, f)
            with open(coordinates_save_path, "wb") as f:
                pickle.dump(coordinates, f)
            logging.info(
                "Saved features to %s, metadata to %s, coordinates to %s",
                feature_save_path,
                metadata_save_path,
                coordinates_save_path,
            )

        if features.shape[1] != args.feature_dim:
            raise ValueError(f"Feature dim mismatch: {features.shape[1]} != {args.feature_dim}")

        index = faiss.IndexFlatIP(args.feature_dim)
        index.add(features)

        return {
            "index": index,
            "metadata": metadata,
            "coordinates": coordinates,
            "db_dir": db_dir,
            "thumb_dir": thumb_dir,
        }

    def _build_shard(
        self,
        args,
        image_encoder,
        thumb_dir: str,
        indices: List[int],
        shard_dir: str,
        rank: int,
        tar_paths_override: List[str] = None,
        show_tar_progress_override: bool = None,
    ):
        tar_paths = tar_paths_override if tar_paths_override is not None else self._find_tar_paths(thumb_dir)
        use_path_inputs = getattr(image_encoder, "use_path_inputs", False)
        if tar_paths:
            if use_path_inputs:
                raise ValueError("WDS thumbnails do not support path-based encoders.")
            if indices:
                raise ValueError("WDS thumbnails do not support index-based sharding.")
            if show_tar_progress_override is None:
                show_tar_progress = os.getenv("WDS_TAR_PROGRESS", "1").strip() == "1"
            else:
                show_tar_progress = show_tar_progress_override
            loader_workers = 0 if show_tar_progress else args.workers
            target_dataset = MarsBenchmarkWDSDataset(
                tar_paths,
                transform=image_encoder.get_processor(),
                show_tar_progress=show_tar_progress,
                progress_desc="Rank 0 tar progress" if show_tar_progress else None,
                progress_position=0,
            )
            subset = target_dataset
        else:
            target_dataset = MarsBenchmarkDataset(
                thumb_dir,
                transform=image_encoder.get_processor(),
                return_path=use_path_inputs,
            )
            subset = Subset(target_dataset, indices)
            loader_workers = args.workers
        target_loader = DataLoader(
            subset,
            batch_size=args.batch_size_database,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=getattr(image_encoder, "collate_fn", None),
        )

        features = []
        metadata = []
        coordinates = []

        for images, image_names in tqdm(target_loader, desc=f"Rank {rank} building shard", unit="batch"):
            image_features = image_encoder.encode_image(images)
            image_features = image_features.cpu().numpy()

            features.append(image_features)
            metadata.extend(image_names)
            batch_coordinates = [_extract_coordinates(name) for name in image_names]
            coordinates.extend(batch_coordinates)

        if features:
            features = np.concatenate(features, axis=0).astype("float32")
            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = features.shape[1]
        else:
            dim = getattr(args, "feature_dim", None) or 0
            features = np.empty((0, dim), dtype="float32")

        shard_paths = {
            "features": os.path.join(shard_dir, f"features_rank{rank}.npy"),
            "metadata": os.path.join(shard_dir, f"metadata_rank{rank}.pkl"),
            "coordinates": os.path.join(shard_dir, f"coordinates_rank{rank}.pkl"),
        }

        np.save(shard_paths["features"], features)
        with open(shard_paths["metadata"], "wb") as f:
            pickle.dump(metadata, f)
        with open(shard_paths["coordinates"], "wb") as f:
            pickle.dump(coordinates, f)
        return shard_paths

    def build_distributed(self, args, image_encoder, delta: float, rank: int, world_size: int) -> Dict:
            import time  # Use time for sleeping during polling
            db_dir, thumb_dir = self._resolve_paths(args, delta)
            shard_dir = os.path.join(db_dir, "shards")
            os.makedirs(shard_dir, exist_ok=True)
            tar_paths = self._find_tar_paths(thumb_dir)

            feature_save_path = os.path.join(db_dir, "features.npy")
            metadata_save_path = os.path.join(db_dir, "metadata.pkl")
            coordinates_save_path = os.path.join(db_dir, "coordinates.pkl")

            # --- 0) All ranks agree on whether the DB exists ---
            local_exists = int(
                os.path.exists(feature_save_path)
                and os.path.exists(metadata_save_path)
                and os.path.exists(coordinates_save_path)
            )

            if torch.cuda.is_available():
                dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                dev = torch.device("cpu")

            exists = torch.tensor(local_exists, device=dev, dtype=torch.int32)
            dist.broadcast(exists, src=0)
            exists = int(exists.item())

            if exists:
                # ... (unchanged code to load existing DB) ...
                out = {}
                if rank == 0:
                    features = np.load(feature_save_path, mmap_mode="r")
                    with open(metadata_save_path, "rb") as f:
                        metadata = pickle.load(f)
                    with open(coordinates_save_path, "rb") as f:
                        coordinates = pickle.load(f)
                    if getattr(args, "feature_dim", None) is None:
                        args.feature_dim = features.shape[1]
                    index = faiss.IndexFlatIP(args.feature_dim)
                    bs = 200000
                    for i in range(0, features.shape[0], bs):
                        index.add(np.asarray(features[i:i+bs]))
                    out = {"index": index, "metadata": metadata, "coordinates": coordinates,
                        "db_dir": db_dir, "thumb_dir": thumb_dir}
                _barrier()
                return out

            # --- 1) Each rank builds its own shard ---
            if tar_paths:
                if getattr(image_encoder, "use_path_inputs", False):
                    raise ValueError("WDS thumbnails do not support path-based encoders.")
                # Shard tar files across ranks.
                tar_paths = tar_paths[rank::world_size]
                indices = []
                show_tar_progress = os.getenv("WDS_TAR_PROGRESS", "1").strip() == "1" and rank == 0
            else:
                dataset = MarsBenchmarkDataset(thumb_dir, transform=image_encoder.get_processor())
                indices = list(range(rank, len(dataset), world_size))
            self._build_shard(
                args,
                image_encoder,
                thumb_dir,
                indices,
                shard_dir,
                rank,
                tar_paths_override=tar_paths if tar_paths else None,
                show_tar_progress_override=show_tar_progress if tar_paths else None,
            )

            # After shard build, write a local flag so rank 0 can poll all shards and avoid barrier timeouts.
            shard_done = os.path.join(shard_dir, f"shard_done_rank{rank}.flag")
            with open(shard_done, "w") as f:
                f.write("done")
            logging.info(f"Rank {rank} shard done, waiting for merge.")

            # --- 2) Rank 0 merge (critical change area) ---
            out = {}
            
            # Simple flag file to notify other ranks
            flag_file = os.path.join(shard_dir, "merge_done.flag")

            if rank == 0:
                try:
                    # Wait for all shard files to be ready to avoid barrier timeout from a slow rank
                    all_done = [os.path.join(shard_dir, f"shard_done_rank{r}.flag") for r in range(world_size)]
                    while not all(os.path.exists(p) for p in all_done):
                        time.sleep(10)

                    logging.info("Rank 0 starting merge...")
                    feature_paths = [os.path.join(shard_dir, f"features_rank{r}.npy") for r in range(world_size)]
                    metadata_paths = [os.path.join(shard_dir, f"metadata_rank{r}.pkl") for r in range(world_size)]
                    coord_paths = [os.path.join(shard_dir, f"coordinates_rank{r}.pkl") for r in range(world_size)]

                    # Compute total rows
                    shard_ns = []
                    shard_dim = getattr(args, "feature_dim", None)
                    for p in feature_paths:
                        if os.path.exists(p):
                            # Read only the shape (no data), which is very fast
                            arr_shape = np.load(p, mmap_mode="r").shape
                            shard_ns.append(arr_shape[0])
                            if shard_dim is None and arr_shape[0] > 0:
                                shard_dim = arr_shape[1]
                        else:
                            shard_ns.append(0)

                    total_rows = sum(shard_ns)
                    if shard_dim is None:
                        shard_dim = args.feature_dim
                    args.feature_dim = shard_dim
                    D = shard_dim

                    # Start writing the large file
                    final = open_memmap(feature_save_path, mode="w+", dtype="float32", shape=(total_rows, D))
                    off = 0
                    for i, (p, n) in enumerate(zip(feature_paths, shard_ns)):
                        if n == 0:
                            continue
                        # Read and write each shard
                        part = np.load(p, mmap_mode="r")
                        final[off:off+n] = part
                        off += n
                        # Optimization: release memory after each shard
                        del part 
                    del final  # Flush to disk

                    # Merge metadata
                    metadata, coordinates = [], []
                    for mp, cp in zip(metadata_paths, coord_paths):
                        if os.path.exists(mp):
                            with open(mp, "rb") as f:
                                metadata.extend(pickle.load(f))
                        if os.path.exists(cp):
                            with open(cp, "rb") as f:
                                coordinates.extend(pickle.load(f))

                    with open(metadata_save_path, "wb") as f:
                        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(coordinates_save_path, "wb") as f:
                        pickle.dump(coordinates, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # Build index
                    features = np.load(feature_save_path, mmap_mode="r")
                    index = faiss.IndexFlatIP(D)
                    bs = 200000
                    for i in range(0, total_rows, bs):
                        index.add(np.asarray(features[i:i+bs]))

                    # Clean up temporary files
                    for p in feature_paths + metadata_paths + coord_paths:
                        if os.path.exists(p):
                            os.remove(p)
                    
                    out = {"index": index, "metadata": metadata, "coordinates": coordinates,
                        "db_dir": db_dir, "thumb_dir": thumb_dir}
                    
                    # Critical step: after rank 0 finishes, write a flag file
                    with open(flag_file, "w") as f:
                        f.write("done")
                    logging.info("Rank 0 merge finished.")

                except Exception as e:
                    logging.error(f"Rank 0 merge failed: {e}")
                    # Re-raise even on failure to prevent other processes from deadlocking
                    raise e
            
            else:
                # Critical change: ranks 1-7 poll a file instead of using barrier directly,
                # so they sleep and release CPU resources to rank 0.
                logging.info(f"Rank {rank} waiting for Rank 0 to merge...")
                while not os.path.exists(flag_file):
                    time.sleep(2)  # Poll every 2 seconds with minimal CPU usage
                logging.info(f"Rank {rank} detected merge done.")

            # Barrier here ensures consistent state; it won't deadlock because rank 0 is done
            _barrier()
            
            # Rank 0 cleans up the flag files
            if rank == 0:
                try:
                    for r in range(world_size):
                        done_flag = os.path.join(shard_dir, f"shard_done_rank{r}.flag")
                        if os.path.exists(done_flag):
                            os.remove(done_flag)
                    os.remove(flag_file)
                    os.rmdir(shard_dir)
                except OSError:
                    pass

            return out

def _barrier():
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()
