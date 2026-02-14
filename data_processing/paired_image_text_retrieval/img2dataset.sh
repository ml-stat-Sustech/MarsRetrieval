#!/usr/bin/env bash
INPUT_TSV=data/paired_image_text_retrieval/dataset/mars_vl_pairs.tsv
OUTPUT_DIR=data/paired_image_text_retrieval/dataset
PROCESSES=16
THREADS=8
IMAGE_SIZE=512

img2dataset \
  --url_list "$INPUT_TSV" \
  --input_format "tsv" \
  --url_col "url" \
  --save_additional_columns '["ori_caption","refined_caption"]' \
  --output_format "webdataset" \
  --output_folder "$OUTPUT_DIR" \
  --number_sample_per_shard 1000 \
  --processes_count 16 \
  --thread_count 8 \
  --image_size 512 \
  --resize_mode center_crop \
  --timeout 10 \
  --retries 3 \
