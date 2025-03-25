import pandas as pd
import pathlib
import tacotoolbox
import tacoreader
import rasterio as rio
import datetime
import ee

# Authenticate and initialize Google Earth Engine
ee.Authenticate()
ee.Initialize(project="ee-samuelsuperresolution")

# Define the root directory for dataset
ROOT_DIR = pathlib.Path("/data/USERS/shollend/taco_example")
table = pd.read_csv(ROOT_DIR / "metadata_updated.csv")

# Iterate over rows of the metadata table to process each image
for i, row in table.iterrows():
    
    # Print progress every 100 rows
    if i % 100 == 0:
        print(f"Processing {i}/{len(table)}")

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
    profile_mask = rio.open(row["hr_mask_path"]).profile
    sample_mask = tacotoolbox.tortilla.datamodel.Sample(
        id="mask",
        path=row["hr_mask_path"],
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
        stac_data={
            "crs": "EPSG:" + str(profile_lr_harm["crs"].to_epsg()),
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

# Process tortilla files and append corresponding metadata
sample_tortillas = []

for index, row in table.iterrows():
    
    # Print progress every 100 rows
    if index % 100 == 0:
        print(f"Processing {index}/{len(table)}")

    # Load tortilla data for each row
    sample_data = tacoreader.load(row["tortilla_path"])
    sample_data = sample_data.iloc[1]

    # Extract 'dist_' and 'count_' columns for additional metadata
    dist_count_dict = {col: row[col] for col in row.index if col.startswith(('dist_', 'count_'))}

    # Create a sample for the tortilla data
    sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
        id=row["image_id"].split(".")[0],
        path=row["tortilla_path"],
        file_format="TORTILLA",
        stac_data={
            "crs": sample_data["stac:crs"],
            "geotransform": sample_data["stac:geotransform"],
            "raster_shape": sample_data["stac:raster_shape"],
            "centroid": sample_data["stac:centroid"],
            "time_start": sample_data["stac:time_start"],
            "time_end": sample_data["stac:time_end"],
        },
        days_diff=row["abs_days_diff"],
        corine=row["corine"],
        in_austria=row["in_austria"],
        archivnr=row["ARCHIVNR"],
        scale_factor=4,
        **dist_count_dict
    )    
    sample_tortillas.append(sample_tortilla)

# Create a collection of all tortilla samples
samples = tacotoolbox.tortilla.datamodel.Samples(
    samples=sample_tortillas
)

# Add RAI metadata to footer (used for further data processing)
samples_obj = samples.include_rai_metadata(
    sample_footprint=1280, # extension in meters
    cache=False,  # Set to True for caching
    quiet=False  # Set to True to suppress the progress bar
)

# Create a collection object with metadata for the dataset
collection_object = tacotoolbox.datamodel.Collection(
    id="sen2austria",
    title="a",  # Update title accordingly
    dataset_version="1.0.0", # Update version accordingly
    description="a",  # Update description accordingly
    licenses=["cc-by-4.0"], 
    extent={
        "spatial": [[-180.0, -90.0, 180.0, 90.0]],  # Define spatial extent
        "temporal": [["2018-01-01T00:00:00Z", "2025-01-17T00:00:00Z"]]  # Define temporal extent
    },
    providers=[{
        "name": "a",  # Update provider name
        "roles": ["host"],
        "links": [{"href": "", "rel": "source", "type": "text/html"}],
    }],
    keywords=["remote-sensing", "super-resolution", "deep-learning", "sentinel-2"],
    task="super-resolution",
    curators=[{
        "name": "Julio Contreras",
        "organization": "Image & Signal Processing",
        "email": ["julio.contreras@uv.es"],
        "links": [{"href": "https://juliocontrerash.github.io/", "rel": "homepage", "type": "text/html"}],
    }],
    split_strategy="none", 
    discuss_link={"href": "", "rel": "source", "type": "text/html"},
    raw_link={"href": "", "rel": "source", "type": "text/html"},
    optical_data={"sensor": "sentinel2msi"},
    labels={"label_classes": [], "label_description": ""},
    scientific={
        "doi": "",
        "citation": "a",  # Update citation
        "summary": "",
        "publications": [{
            "doi": "",
            "citation": "a",  # Update citation
            "summary": "",
        }]
    },
)

# Get the path of the tortilla file and create the directory if needed
full_path = table["tortilla_path"].iloc[0]
directory = pathlib.Path(full_path).parts[0]

# Generate the final output file using the samples and collection objects
output_file = tacotoolbox.create(
    samples=samples_obj,
    collection=collection_object,
    output=directory + "/" + "austria.taco"
)
