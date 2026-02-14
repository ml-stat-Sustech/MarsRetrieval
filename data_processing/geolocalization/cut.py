import os
import re
import math
import random
from osgeo import gdal
from pyproj import CRS, Transformer
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial
from typing import List, Tuple
import numpy as np
import torch


def parse_filename_corner(tif_path: str):
    filename = os.path.basename(tif_path)
    pattern = r"[EN]-?\d+"
    matches = re.findall(pattern, filename.upper())
    if not matches or len(matches) != 2:
        raise ValueError(f"Failed to parse E_xxx_N_xxx pattern from filename: {filename}")

    lon, lat = None, None
    for match in matches:
        if match.startswith("E"):
            lon_val = int(match.lstrip("E-"))
            lon = -lon_val if match.startswith("E-") else lon_val
        elif match.startswith("N"):
            lat_val = int(match.lstrip("N-"))
            lat = -lat_val if match.startswith("N-") else lat_val

    if lon is None or lat is None:
        raise ValueError(f"Failed to parse E_xxx_N_xxx pattern from filename: {filename}")

    return lon, lat


def slice_task(
    cell,
    input_tif,
    output_folder,
    start_lon,
    start_lat,
    delta_deg,
    width_deg,
    height_deg,
    src_wkt,
    decimals,
    progress_queue,
    image_size,
):
    row, col = cell
    min_lon = start_lon + col * delta_deg
    max_lon = min(min_lon + delta_deg, start_lon + width_deg)
    min_lat = start_lat + row * delta_deg
    max_lat = min(min_lat + delta_deg, start_lat + height_deg)

    crs_merc = CRS.from_wkt(src_wkt)
    mars_geo_proj4 = "+proj=longlat +a=3396190 +b=3396190 +no_defs"
    crs_geo = CRS.from_string(mars_geo_proj4)
    transformer = Transformer.from_crs(crs_geo, crs_merc, always_xy=True)

    x_min, y_min = transformer.transform(min_lon, min_lat)
    x_max, y_max = transformer.transform(max_lon, max_lat)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    lon_str = f"{min_lon:.{decimals}f}".replace(".", "_").replace("-", "_")
    lat_str = f"{min_lat:.{decimals}f}".replace(".", "_").replace("-", "_")
    out_name = f"E{lon_str}_N{lat_str}.png"
    out_path = os.path.join(output_folder, out_name)

    width_m = abs(x_max - x_min)
    height_m = abs(y_max - y_min)
    if width_m <= 0 or height_m <= 0:
        progress_queue.put(1)
        return

    width_px = image_size
    aspect = height_m / width_m
    height_px = int(round(width_px * aspect))

    try:
        warp_opts = gdal.WarpOptions(
            format="PNG",
            outputBounds=[x_min, y_min, x_max, y_max],
            width=width_px,
            height=height_px,
            resampleAlg=gdal.GRA_Bilinear,
            dstNodata=0,
        )
        ds_slice = gdal.Warp(out_path, input_tif, options=warp_opts)
        if ds_slice:
            ds_slice = None
        else:
            print(f"[Warning] Crop failed => {out_path}")
    except Exception as e:
        print(f"[Error] Failed to process {out_path}: {str(e)}")
    finally:
        progress_queue.put(1)


def slice_mars_tif_by_lonlat(
    input_tif: str,
    output_folder: str,
    start_lon: float,
    start_lat: float,
    width_deg: float,
    height_deg: float,
    delta_deg: float,
    image_size: int = 512,
):
    gdal.UseExceptions()

    ds = gdal.Open(input_tif)
    if ds is None or ds.GetProjection() is None:
        with open("failed.txt", "a") as f:
            f.write(f"Failed to open file: {input_tif}\n")
        return
    src_wkt = ds.GetProjection()
    ds = None

    n_cols = int(math.ceil(width_deg / delta_deg))
    n_rows = int(math.ceil(height_deg / delta_deg))

    decimals = max(0, int(math.ceil(-math.log10(delta_deg))))

    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    selected_num = int(1 * n_cols * n_rows)

    selected_cells = random.sample(all_cells, selected_num)

    manager = Manager()
    progress_queue = manager.Queue()

    with Pool(processes=min(os.cpu_count(), 32)) as pool:
        with tqdm(total=selected_num, desc="Crop progress", unit="tile") as pbar:
            pool.map_async(
                partial(
                    slice_task,
                    input_tif=input_tif,
                    output_folder=output_folder,
                    start_lon=start_lon,
                    start_lat=start_lat,
                    delta_deg=delta_deg,
                    width_deg=width_deg,
                    height_deg=height_deg,
                    src_wkt=src_wkt,
                    decimals=decimals,
                    progress_queue=progress_queue,
                    image_size=image_size,
                ),
                selected_cells,
            )
            for _ in range(selected_num):
                progress_queue.get()
                pbar.update(1)

    print("All splits completed!")


def collect_tif_files(folder_path: str) -> List[str]:
    tif_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".tif"):
                tif_files.append(os.path.join(root, file))

    tif_files.sort()
    return tif_files


def random_seed(seed=42, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)

        import torch.backends.cudnn as cudnn

        cudnn.deterministic = True
        cudnn.benchmark = False
