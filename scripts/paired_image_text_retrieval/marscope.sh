#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings

TASK_CONFIG=PairedVLRetrieval
MODEL_CONFIG=CLIPMarScope
EXP_NAME=main_exp

MODEL_NAME=ViT-L-14-quickgelu
PRETRAINED=dfn2b
RESUME_POST_TRAINS=(./model/logs/ckpt/ViT-L-14-quickgelu_dfn2b/checkpoints/ckpt.pt)


IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip

for RESUME_POST_TRAIN in "${RESUME_POST_TRAINS[@]}"; do
  echo "Evaluating model with post-training checkpoint: ${RESUME_POST_TRAIN}"

  python main.py \
    --task_config "${TASK_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --query_mode "text" \
    --model_name "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}" \
    --resume_post_train "${RESUME_POST_TRAIN}" \
    --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
    --text_encoder_type "${TEXT_ENCODER_TYPE}"
done