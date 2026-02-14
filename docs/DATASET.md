# Dataset Preparation

All datasets are hosted on Hugging Face:
`https://huggingface.co/collections/claytonwang/marsretrieval`

<!-- If the repos are private, export your token once:
```
export HF_TOKEN=your_token_here
``` -->

---

**Task 1: Paired Image–Text Retrieval**

**Build steps**
1. Export HF dataset to TSV.
```
python data_processing/paired_image_text_retrieval/hf2tsv.py
```
2. Download images and build WDS shards.
```
bash data_processing/paired_image_text_retrieval/img2dataset.sh
```

**Final Layout**
```
$DATA/
  paired_image_text_retrieval/
    dataset/
      mars_vl_pairs.tsv
      000000.tar
      000001.tar
      ...
```

---

**Task 2: Landform Retrieval**

**Build steps**
1. Download HF dataset.
```
python data_processing/landform_retrieval/download_hf_landforms.py
```

**Final Layout**
```
$DATA/
  landform_retrieval/
    dataset/
      data/
        train-00000-of-00002.parquet
        train-00001-of-00002.parquet
```

---

**Task 3: Global Geo-Localization**

**Build steps**
0.  Install GDAL wheel.
```
# GDAL
mkdir -p software
cd software
wget https://github.com/girder/large_image_wheels/raw/wheelhouse/GDAL-3.11.3.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=a101c5c5782daf719722f4d5b9fdcf5149ed2bdee48c2758cf706758ef2b07d2
pip install GDAL-3.10.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
cd ..
```
1. Download HF dataset folders (ground_truth / image_queries / mars_map).
```
python data_processing/geolocalization/download_hf_geolocalization.py
```
2. Download CTX tiles and unzip.
```
bash data_processing/geolocalization/download_ctx_tiles.sh
```
3. Build tiles and thumbnails.
```
python data_processing/geolocalization/build_tiles.py \
  --raw-dir data/global_geolocalization/tiles_raw \
  --dataset-dir data/global_geolocalization/dataset \
  --delta-deg 0.2 \
  --image-size 512 \
  --thumb-size 512 \
  --max-abs-lat 80 \
  --wds-shard-size 1000 \
  --save-format wds
```

**Final Layout**
```
$DATA/
  global_geolocalization/
    dataset/
      ground_truth/
      image_queries/
      mars_map/
      tiles/
        image_size_512_delta_0.2/
          thumb/
            <tif_name>/
              000000.tar
              000001.tar
              ...
```
