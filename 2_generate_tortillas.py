import pandas as pd
import pathlib
import tacotoolbox
import tacoreader
import time
import os
from multiprocessing import Pool
from tqdm import tqdm
import rasterio as rio
import datetime
#import ee

# Authenticate and initialize Google Earth Engine
# ee.Authenticate()
# ee.Initialize(project="ee-samuelsuperresolution")

# Define the root directory for dataset
ROOT_DIR = pathlib.Path("/data/USERS/shollend/taco")
table = pd.read_csv(ROOT_DIR / "metadata_updated.csv")


def _parallel(row: pd.Series) -> None:
    # Print progress every 100 rows
    # Load profiles and metadata for the low-resolution (LR) image
    profile_lr = rio.open(row["lr_s2_path"]).profile
    sample_lr = tacotoolbox.tortilla.datamodel.Sample(
        id="lr",
        path=row["lr_s2_path"],
        file_format="GTiff",
        data_split="train",
        stac_data={
            "crs": "EPSG:" + str(profile_lr["crs"].to_epsg()),
            "geotransform": profile_lr["transform"].to_gdal(),
            "raster_shape": (profile_lr["height"], profile_lr["width"]),
            "time_start": datetime.datetime.strptime(row.time, '%Y-%m-%d'),
            "time_end": datetime.datetime.strptime(row.time, '%Y-%m-%d'),
            "centroid": f"POINT ({row['lon']} {row['lat']})"
        },
        s2_id_gee=row["s2_full_id"],
        cloud_perc=((1 - row["cs_cdf"]) * 100)
    )

    # Load profiles and metadata for the high-resolution (HR) image
    profile_hr = rio.open(row["hr_othofoto_path"]).profile
    sample_hr = tacotoolbox.tortilla.datamodel.Sample(
        id="hr",
        path=row["hr_othofoto_path"],
        file_format="GTiff",
        data_split="train",
        stac_data={
            "crs": "EPSG:" + str(profile_hr["crs"].to_epsg()),
            "geotransform": profile_hr["transform"].to_gdal(),
            "raster_shape": (profile_hr["height"], profile_hr["width"]),
            "time_start": datetime.datetime.strptime(row.ortho_begin_date, '%Y-%m-%d'),
            "time_end": datetime.datetime.strptime(row.ortho_end_date, '%Y-%m-%d'),
            "centroid": f"POINT ({row['lon']} {row['lat']})"
        },
    )

    # Load profiles and metadata for the mask image
    profile_mask = rio.open(row["hr_compressed_mask_path"]).profile
    sample_mask = tacotoolbox.tortilla.datamodel.Sample(
        id="mask",
        path=row["hr_compressed_mask_path"],
        file_format="GTiff",
        data_split="train",
        stac_data={
            "crs": "EPSG:" + str(profile_mask["crs"].to_epsg()),
            "geotransform": profile_mask["transform"].to_gdal(),
            "raster_shape": (profile_mask["height"], profile_mask["width"]),
            "time_start": datetime.datetime.strptime(row.ortho_begin_date, '%Y-%m-%d'),
            "time_end": datetime.datetime.strptime(row.ortho_end_date, '%Y-%m-%d'),
            "centroid": f"POINT ({row['lon']} {row['lat']})"
        },
    )

    # Load profiles and metadata for the harmonized LR and HR images
    profile_lr_harm = rio.open(row["lr_harm_path"]).profile
    sample_lr_harm = tacotoolbox.tortilla.datamodel.Sample(
        id="lr_harm",
        path=row["lr_harm_path"],
        file_format="GTiff",
        data_split="train",
        stac_data={"crs": "EPSG:" + str(profile_lr_harm["crs"].to_epsg()),
                   "geotransform": profile_lr_harm["transform"].to_gdal(),
                   "raster_shape": (profile_lr_harm["height"], profile_lr_harm["width"]),
                   "time_start": datetime.datetime.strptime(row.time, '%Y-%m-%d'),
                   "time_end": datetime.datetime.strptime(row.time, '%Y-%m-%d'),
                   "centroid": f"POINT ({row['lon']} {row['lat']})"
                   },
    )

    # Load profiles and metadata for the harmonized HR image
    profile_hr_harm = rio.open(row["hr_harm_path"]).profile
    sample_hr_harm = tacotoolbox.tortilla.datamodel.Sample(
        id="hr_harm",
        path=row["hr_harm_path"],
        file_format="GTiff",
        data_split="train",
        stac_data={
            "crs": "EPSG:" + str(profile_hr_harm["crs"].to_epsg()),
            "geotransform": profile_hr_harm["transform"].to_gdal(),
            "raster_shape": (profile_hr_harm["height"], profile_hr_harm["width"]),
            "time_start": datetime.datetime.strptime(row.ortho_begin_date, '%Y-%m-%d'),
            "time_end": datetime.datetime.strptime(row.ortho_end_date, '%Y-%m-%d'),
            "centroid": f"POINT ({row['lon']} {row['lat']})"
        },
    )

    # Create a set of samples for each image type (LR, HR, mask, harmonized LR and HR)
    samples = tacotoolbox.tortilla.datamodel.Samples(
        samples=[
            sample_lr,
            sample_hr,
            sample_mask,
            sample_lr_harm,
            sample_hr_harm
        ]
    )

    # Create the tortilla (data object) for this row
    tacotoolbox.tortilla.create(samples, row["tortilla_path"], quiet=True)
    return


def process_parallel():
    print('start')
    start = time.time()
    rows = [row for _, row in table.iterrows()]

    with Pool(processes=os.cpu_count()) as pool:
        #results = pool.map(_parallel, rows)
        res = tqdm(pool.imap_unordered(_parallel, rows), total=len(rows), desc="Processing")

    print(f'took: {round(time.time() - start, 2)}s')

    return


def process_sequential():
    print('start')
    start = time.time()

    for _, row in table.iterrows():
        if _ % 100 == 0:
            print(f"Processing {_}/{len(table)}")
        res = _parallel(row)

    print(f'took: {round(time.time() - start, 2)}s')

    return


process_sequential()