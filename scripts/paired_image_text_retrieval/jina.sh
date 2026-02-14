#!/usr/bin/env bash
# Usage: bash scripts/paired_image_text_retrieval/jina.sh 0,1 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH


TASK_CONFIG=PairedVLRetrieval
MODEL_CONFIG=Jina
EXP_NAME=main_exp

MODEL_NAME=jinaai/jina-embeddings-v4
PRETRAINED=hf
IMAGE_ENCODER_TYPE=jina
TEXT_ENCODER_TYPE=jina

python main.py \
  --task_config "${TASK_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "text" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}"
