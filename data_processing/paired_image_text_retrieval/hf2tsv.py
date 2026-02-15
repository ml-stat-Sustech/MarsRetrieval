#!/usr/bin/env python3
import csv
import os

from datasets import load_dataset

DATASET = "SUSTech/Mars-VL-Pairs"
SPLIT = "train"
OUTPUT = "data/paired_image_text_retrieval/dataset/mars_vl_pairs.tsv"


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ds = load_dataset(DATASET, split=SPLIT, token=token)

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["url", "ori_caption", "refined_caption", "key"])
        count = 0
        for row in ds:
            url = row.get("image_url")
            ori_caption = row.get("ori_caption")
            refined_caption = row.get("refined_caption")
            key = row.get("key", "")
            if not url or not ori_caption or not refined_caption:
                continue
            writer.writerow([url, ori_caption, refined_caption, key])
            count += 1

    print(f"Wrote {count} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
