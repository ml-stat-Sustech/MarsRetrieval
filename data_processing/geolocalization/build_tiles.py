#!/usr/bin/env python3
import argparse
import os
import shutil
import tarfile
import time
from tqdm import tqdm

from cut import slice_mars_tif_by_lonlat, parse_filename_corner, collect_tif_files, random_seed
from thumb_gen import generate_thumbnails


def main():
    parser = argparse.ArgumentParser(description="Build geolocalization tiles and thumbnails from CTX TIFFs.")
    parser.add_argument("--raw-dir", type=str, default="data/global_geolocalization/tiles_raw", help="Folder with .tif files")
    parser.add_argument("--dataset-dir", type=str, default="data/global_geolocalization/dataset", help="Dataset root")
    parser.add_argument("--delta-deg", type=float, default=0.2, help="Tile size in degrees")
    parser.add_argument("--max-abs-lat", type=float, default=80, help="Skip tiles beyond this abs latitude")
    parser.add_argument("--image-size", type=int, default=512, help="Output PNG width in pixels")
    parser.add_argument("--thumb-size", type=int, default=512, help="Thumbnail size in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--keep-cache", action="store_true", help="Keep intermediate cache folders")
    parser.add_argument("--thumb-workers", type=int, default=32, help="Thumbnail workers")
    parser.add_argument(
        "--save-format",
        type=str,
        default="wds",
        choices=["wds", "png"],
        help="Output format for thumbnails: wds (tar shards) or png files",
    )
    parser.add_argument("--wds-shard-size", type=int, default=1000, help="Images per WDS shard")
    args = parser.parse_args()

    random_seed(args.seed)

    tiles_root = os.path.join(
        args.dataset_dir,
        "tiles",
        f"image_size_{args.image_size}_delta_{args.delta_deg}",
    )
    cache_dir = os.path.join(tiles_root, "cache")
    thumb_root = os.path.join(tiles_root, "thumb")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(thumb_root, exist_ok=True)

    tif_files = collect_tif_files(args.raw_dir)
    print(f"Found {len(tif_files)} .tif files")

    for input_tif in tqdm(tif_files, desc="Processing TIF files"):
        file_name = os.path.basename(input_tif)
        try:
            start_lon, start_lat = parse_filename_corner(input_tif)
        except ValueError as e:
            print(f"[Skip] {file_name}: {e}")
            continue

        t_lat = start_lat + 4 if start_lat < 0 else start_lat
        if abs(t_lat) >= args.max_abs_lat:
            print(f"[Skip] {file_name} (latitude={start_lat}) exceeds threshold {args.max_abs_lat}")
            continue

        output_folder = os.path.join(cache_dir, os.path.splitext(file_name)[0])
        os.makedirs(output_folder, exist_ok=True)

        slice_mars_tif_by_lonlat(
            input_tif=input_tif,
            output_folder=output_folder,
            start_lon=start_lon,
            start_lat=start_lat,
            width_deg=4.0,
            height_deg=4.0,
            delta_deg=args.delta_deg,
            image_size=args.image_size,
        )

        thumb_dir = os.path.join(thumb_root, os.path.splitext(file_name)[0])
        generate_thumbnails(
            output_folder,
            thumb_dir,
            thumb_size=args.thumb_size,
            args=None,
            max_workers=args.thumb_workers,
        )

        if args.save_format == "wds":
            shard_dir = os.path.join(thumb_root, os.path.splitext(file_name)[0])
            os.makedirs(shard_dir, exist_ok=True)
            try:
                png_files = [f for f in sorted(os.listdir(thumb_dir)) if f.lower().endswith(".png")]
                for i in range(0, len(png_files), args.wds_shard_size):
                    shard_idx = i // args.wds_shard_size
                    tar_path = os.path.join(shard_dir, f"{shard_idx:06d}.tar")
                    with tarfile.open(tar_path, "a") as tar:
                        for fname in png_files[i : i + args.wds_shard_size]:
                            full_path = os.path.join(thumb_dir, fname)
                            tar.add(full_path, arcname=fname)
                for fname in png_files:
                    try:
                        os.remove(os.path.join(thumb_dir, fname))
                    except FileNotFoundError:
                        pass
            except Exception as e:
                print(f"Failed to archive {thumb_dir}: {e}")

        if not args.keep_cache:
            try:
                start_time = time.time()
                shutil.rmtree(output_folder)
                end_time = time.time()
                print(f"Deleted temp folder: {output_folder} (took {end_time - start_time:.2f} s)")
            except Exception as e:
                print(f"Failed to delete {output_folder}: {e}")


if __name__ == "__main__":
    main()
