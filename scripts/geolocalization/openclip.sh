export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings

TASK_CONFIG=GlobalGeoLocalization
MODEL_CONFIG=OpenCLIP
EXP_NAME=main_exp

# query settings
QUERY_MODES=(image text)
QUERY_TEXTS=(alluvial_fans glacier-like_form landslides pitted_cones yardangs)


# model (OpenCLIP by default)
MODEL_NAMES=(
  # ViT-L-14-quickgelu
  # ViT-L-16-SigLIP-384
  # ViT-L-16-SigLIP2-512
  PE-Core-L-14-336
)
PRETRAINEDS=(
  # dfn2b
  # hf-hub:timm/ViT-L-16-SigLIP-384
  # hf-hub:timm/ViT-L-16-SigLIP2-512
  hf-hub:timm/PE-Core-L-14-336
)

IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip


NPROC=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr -cd ',' | wc -c) + 1 ))

for IDX in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$IDX]}"
  PRETRAINED="${PRETRAINEDS[$IDX]}"

  # optional distributed DB build (skips if DB already exists)
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node=${NPROC} build_geo_db.py \
      --task_config "${TASK_CONFIG}" \
      --model_config "${MODEL_CONFIG}" \
      --exp_name "${EXP_NAME}" \
      --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
      --text_encoder_type "none" \
      --model_name "${MODEL_NAME}" \
      --pretrained "${PRETRAINED}"
  fi

  # run retrieval
  for QUERY_MODE in "${QUERY_MODES[@]}"; do
    for QUERY_TEXT in "${QUERY_TEXTS[@]}"; do
      python main.py \
      --task_config "${TASK_CONFIG}" \
      --model_config "${MODEL_CONFIG}" \
      --exp_name "${EXP_NAME}" \
      --query_mode "${QUERY_MODE}" \
      --query_text "${QUERY_TEXT}" \
      --model_name "${MODEL_NAME}" \
      --pretrained "${PRETRAINED}" \
      --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
      --text_encoder_type "${TEXT_ENCODER_TYPE}"
    done
  done
done
