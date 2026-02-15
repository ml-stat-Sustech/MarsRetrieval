#!/usr/bin/env python3
from huggingface_hub import snapshot_download

DATASET = "SUSTech/Mars-Global-Geolocalization"
OUTPUT_DIR = "data/global_geolocalization/dataset"


def main():
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False,
        token=None,  # reads HF_TOKEN if set
    )
    print(f"Downloaded {DATASET} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
