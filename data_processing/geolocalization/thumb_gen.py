import os
import time
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from osgeo import gdal
from concurrent.futures import ProcessPoolExecutor, as_completed


def is_master(args=None):
    return True


def process_single_thumbnail(img_path, thumb_dir, thumb_size):
    try:
        img_name = os.path.basename(img_path)
        thumb_name = os.path.splitext(img_name)[0] + ".png"
        thumb_path = os.path.join(thumb_dir, thumb_name)

        if img_path.lower().endswith((".tif", ".tiff")):
            gdal.UseExceptions()
            dataset = gdal.Open(img_path)
            if dataset is None:
                return None, f"GDAL failed to open {img_path}"

            img_array = dataset.ReadAsArray()
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=0)
            if img_array.shape[0] > 3:
                img_array = img_array[:3, :, :]
            img_array = np.transpose(img_array, (1, 2, 0))

            if img_array.dtype != np.uint8:
                min_val, max_val = np.min(img_array), np.max(img_array)
                if max_val > min_val:
                    img_array = (255 * (img_array - min_val) / (max_val - min_val)).astype(np.uint8)
                else:
                    img_array = np.zeros(img_array.shape, dtype=np.uint8)

            img = Image.fromarray(img_array).convert("RGB")
            dataset = None
        else:
            img = Image.open(img_path).convert("RGB")

        thumbnail = img.resize((thumb_size, thumb_size), Image.Resampling.BICUBIC)
        thumbnail.save(thumb_path, "PNG", optimize=True)
        return thumb_path, None

    except UnidentifiedImageError:
        return None, f"Unrecognized image file: {img_path}"
    except Exception as e:
        return None, f"Error processing {img_path}: {e}"


def generate_thumbnails(image_dir, save_dir, thumb_size=128, args=None, max_workers=None):
    if not is_master(args):
        return

    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg"))
    ]
    thumb_dir = os.path.join(save_dir)
    os.makedirs(thumb_dir, exist_ok=True)

    workers = max_workers or os.cpu_count() or 8
    print(f"Using {workers} processes to generate thumbnails for {len(image_paths)} images")

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(process_single_thumbnail, path, thumb_dir, thumb_size): path
            for path in image_paths
        }

        for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="Generating thumbnails"):
            path = future_to_path[future]
            try:
                thumb_path, error_message = future.result()
                if error_message:
                    print(f"\nWarning: {error_message}")
            except Exception as e:
                print(f"\nException while processing image {path}: {e}")

    total_time = time.perf_counter() - start_time
    avg_time_per_image = total_time / len(image_paths) if image_paths else 0
    print(f"\nTotal time for thumbnail generation: {total_time:.2f} s")
    print(f"Average time per image: {avg_time_per_image * 1000:.4f} ms")
