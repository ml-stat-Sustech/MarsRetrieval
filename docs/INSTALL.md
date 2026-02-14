# Environment setup

**Setup conda environment (recommended).**
```bash
# Create a conda environment
conda create -y -n retrieval python=3.12
# Activate the environment
conda activate retrieval

# Install requirements
pip install -r requirements.txt

# Optional (flash-attention)
mkdir -p software
cd software
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
cd ..
```

**Model dependency.**
```bash
mkdir -p third_party
cd third_party

# for vlm2vec and b3_qwen
git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git vlm2vec

# for ops_mm
git clone https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B ops_mm_embedding

# for qwen3_vl_embedding
mkdir -p qwen3_vl_embedding
cd qwen3_vl_embedding
wget https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/scripts/qwen3_vl_embedding.py

# for gme
pip install transformers==4.51.3
```
Note: the project default is `transformers==4.57.3` (see `requirements.txt`). Only install `4.51.3` if you specifically run `gme`.
