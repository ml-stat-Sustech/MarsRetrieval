#!/usr/bin/env bash
BASE_URL="https://murray-lab.caltech.edu/CTX/V01/tiles/"
DOWNLOAD_DIR="data/global_geolocalization/tiles_raw_zips"
UNZIP_DIR="data/global_geolocalization/tiles_raw"

mkdir -p "$DOWNLOAD_DIR" "$UNZIP_DIR"

# download all zip files
wget -c -r -l1 -np -nd -A "*.zip" "$BASE_URL" -P "$DOWNLOAD_DIR"

# unzip into a single folder
for z in "$DOWNLOAD_DIR"/*.zip; do
  [ -f "$z" ] || continue
  unzip -n "$z" -d "$UNZIP_DIR"
done

# keep only .tif files
find "$UNZIP_DIR" -type f ! -name "*.tif" -delete
