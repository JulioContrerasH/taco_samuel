from utils_histogram import *
import pathlib
import pandas as pd
import rasterio as rio
import torch
from skimage.exposure import match_histograms

ROOT_DIR = pathlib.Path("/data/USERS/shollend/taco_example")
table = pd.read_csv(ROOT_DIR / "metadata.csv")

# Generate file paths for each image type (high-resolution, low-resolution, etc.)
table["hr_mask_path"] = ROOT_DIR / "hr_mask" / ("HR_mask_" + table["image_id"])  
table["hr_othofoto_path"] = ROOT_DIR / "hr_orthofoto" / ("HR_ortho_" + table["image_id"])
table["lr_s2_path"] = ROOT_DIR / "lr_s2" /  ("S2_" + table["image_id"])
table["lr_harm_path"] = ROOT_DIR / "lr_harm" /  ("lr_harm_" + table["image_id"])
table["hr_harm_path"] = ROOT_DIR / "hr_harm" /  ("hr_harm_" + table["image_id"])
table["tortilla_path"] = ROOT_DIR / "tortilla" /  (table["image_id"].str.split(".").str[0] + ".tortilla")

# Ensure the necessary directories exist
parent_lr = table["lr_harm_path"].iloc[0].parent
parent_lr.mkdir(exist_ok=True)
parent_hr = table["hr_harm_path"].iloc[0].parent
parent_hr.mkdir(exist_ok=True)
parent_tortilla = table["tortilla_path"].iloc[0].parent
parent_tortilla.mkdir(exist_ok=True)

low_cors = []

# Iterate over each row in the metadata table
for i, row in table.iterrows():
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
    low_cors.append(low_cor)

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

# Add the low correlation values to the table and save it to a new CSV file
table["low_cor"] = low_cors
table.to_csv(ROOT_DIR / "metadata_updated.csv")
