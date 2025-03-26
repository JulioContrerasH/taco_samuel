import numpy as np

from utils_histogram import *
import pathlib
import time
import os
from multiprocessing import Pool
import rasterio as rio
import torch
import pandas as pd
from tqdm import tqdm
from skimage.exposure import match_histograms

#np.seterr(all="raise")  # Convert warnings to exceptions

ROOT_DIR = pathlib.Path("/data/USERS/shollend/taco_example")
table = pd.read_csv(ROOT_DIR / "metadata.csv")

# Generate file paths for each image type (high-resolution, low-resolution, etc.)
table["hr_mask_path"] = ROOT_DIR / "hr_mask" / ("HR_mask_" + table["image_id"])
table["hr_compressed_mask_path"] = ROOT_DIR / "hr_compressed_mask" / ("HR_mask_" + table["image_id"])
table["hr_othofoto_path"] = ROOT_DIR / "hr_orthofoto" / ("HR_ortho_" + table["image_id"])
table["hr_compressed_othofoto_path"] = ROOT_DIR / "hr_compressed_orthofoto" / ("HR_ortho_" + table["image_id"])
table["lr_s2_path"] = ROOT_DIR / "lr_s2" /  ("S2_" + table["image_id"])
table["lr_harm_path"] = ROOT_DIR / "lr_harm" /  ("lr_harm_" + table["image_id"])
table["hr_harm_path"] = ROOT_DIR / "hr_harm" /  ("hr_harm_" + table["image_id"])
table["tortilla_path"] = ROOT_DIR / "tortilla" /  (table["image_id"].str.split(".").str[0] + ".tortilla")

# Ensure the necessary directories exist
parent_lr = table["lr_harm_path"].iloc[0].parent
parent_lr.mkdir(exist_ok=True)
parent_hr_mask = table["hr_compressed_mask_path"].iloc[0].parent
parent_hr_mask.mkdir(exist_ok=True)
parent_hr_compressed = table["hr_compressed_othofoto_path"].iloc[0].parent
parent_hr_compressed.mkdir(exist_ok=True)
parent_hr = table["hr_harm_path"].iloc[0].parent
parent_hr.mkdir(exist_ok=True)
parent_tortilla = table["tortilla_path"].iloc[0].parent
parent_tortilla.mkdir(exist_ok=True)


def _parallel(row: pd.Series) -> np.array:
    # Iterate over each row in the metadata table
    # compress HR mask
    with rio.open(row["hr_mask_path"]) as src_hr_mask:
        metadata_hr_mask = src_hr_mask.meta
        src_hr_mask_data = src_hr_mask.read()

    metadata_hr_mask.update(
        dtype=rio.uint8,
        compress="zstd",
        zstd_level=13,
        interleave="band",
        tiled=True,
        blockxsize=128,
        blockysize=128,
        # discard_lsb removes class labels111
        #discard_lsb=2
    )

    with rio.open(row["hr_compressed_mask_path"], "w", **metadata_hr_mask) as dst:
        dst.write(src_hr_mask_data)

    # Open and read high-resolution (HR) and low-resolution (LR) images
    with rio.open(row["hr_othofoto_path"]) as src_hr, rio.open(row["lr_s2_path"]) as src_lr:
        metadata_hr = src_hr.meta
        hr = src_hr.read() / 255  # Normalize HR image
        metadata_lr = src_lr.meta
        lr = src_lr.read([4, 3, 2, 8]) / 10_000  # Normalize selected LR bands

    # Match histograms between HR and LR images
    hrharm = match_histograms(hr, lr, channel_axis=0)

    # Degrade HRharm to LRharm using bilinear interpolation
    lrharm = torch.nn.functional.interpolate(
        torch.from_numpy(hrharm).unsqueeze(0),
        scale_factor=0.25,
        mode="bilinear",
        antialias=True
    ).squeeze().numpy()

    # Compute block-wise correlation between LR and LRharm
    kernel_size = 32
    corr = fast_block_correlation(lr, lrharm, block_size=kernel_size)

    # Report the 10th percentile of the correlation (low correlation)
    low_cor = np.quantile(corr, 0.10)  # This value is added to the dataset
    row["low_corr"] = low_cor

    # Save the HRharm image with updated metadata
    metadata_hrharm = metadata_hr.copy()
    metadata_hrharm.update(
        dtype=rio.uint16,
        compress="zstd",
        zstd_level=13,
        interleave="band",
        tiled=True,
        blockxsize=128,
        blockysize=128,
        discard_lsb=2
    )

    with rio.open(row["hr_harm_path"], "w", **metadata_hrharm) as dst:
        dst.write((hrharm * 10_000).round().astype(rio.uint16))

    # Save the LRharm image with updated metadata
    metadata_lrharm = metadata_lr.copy()
    metadata_lrharm.update(
        dtype=rio.uint16,
        compress="zstd",
        count=4,
        zstd_level=13,
        interleave="band",
        tiled=True,
        blockxsize=32,
        blockysize=32,
        discard_lsb=2
    )

    with rio.open(row["lr_harm_path"], "w", **metadata_lrharm) as dst:
        dst.write((lrharm * 10_000).round().astype(rio.uint16))

    return row


def process_parallel():
    print('start')
    start = time.time()
    rows = [row for _, row in table.iterrows()]

    with Pool(processes=os.cpu_count()) as pool:
        #results = pool.map(_parallel, rows)
        results = list(tqdm(pool.imap_unordered(_parallel, rows), total=len(rows), desc="Processing"))

    new_df = pd.DataFrame(results)

    print(f'took: {round(time.time() - start, 2)}s')

    return new_df


def process_sequential():
    print('start')
    start = time.time()

    results = []
    for _, row in table.iterrows():
        results.append(_parallel(row))

    new_df = pd.DataFrame(results)

    print(f'took: {round(time.time() - start, 2)}s')

    return new_df


# Add the low correlation values to the table and save it to a new CSV file
table_new = process_parallel()
# table_new.to_csv(ROOT_DIR / "metadata_updated.csv")
