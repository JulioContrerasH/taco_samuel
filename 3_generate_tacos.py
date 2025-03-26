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
ROOT_DIR = pathlib.Path("/data/USERS/shollend/taco")
table = pd.read_csv(ROOT_DIR / "metadata_updated.csv")

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
    dist_count_dict = {col: row[col] for col in row.index if col.startswith(('dist_'))}

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
        s2_full_id=row["s2_full_id"],
        s2_tile_time=row["time"],
        cs_cdf=row["cs_cdf"],
        low_corr=row["low_corr"],
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
description="""

A dataset containing 52105 tiles of paired Sentinel-2 multispectral images, RGBNIR Orthofoto imagery, and cadastral ground truth information covering all of Austria for validating super-resolution (SR) algorithms. 
Each pair consists of a Sentinel-2 image at 10 m resolution, a spatially and temporally aligned Orthofoto image at 2.5 m resolution (resampled from original 0.2 m), and a corresponding ground truth mask with cadastral classes rasterized to similar resolution and extent as the Orthofoto.

**Sentinel-2 MSI:** Sentinel-2 is a twin-satellite mission (2A/2B) providing optical imagery with 13 spectral bands spanning visible, near-infrared (VNIR) and shortwave-infrared (SWIR) wavelengths. The Multispectral Instrument (MSI) samples four bands at 10 m, six bands at 20 m, and three bands at 60 m spatial resolution. Sentinel-2’s bands cover 443 nm (coastal aerosol) to 2202 nm (SWIR), supporting applications in vegetation monitoring, water resources, land cover and more. The mission offers global land coverage every ~5 days (with both satellites) and a free, open data policy. In this dataset, Sentinel-2 Level-2A surface reflectance images are used as the **low-resolution (LR)** input. Only images with a cloud score >= 0.8 were selected.


**Orthofoto:** The Orthofotos are captured by each Austrian state indivudally every three years and are aggregated, orthorectifies, and provided by the Austrian land surveying agency Bundesamt für Eich- und Vermessugnswesen (BEV). The images are captured during the summer months (May to September) during Austrias peak agricultural season, but not at the same time, as each state follows its own three year cycle (see Figure below). Images have a high resolution of 0.2 m along four spectral bands: Red, Green, Blue, Near Infrared, which have been resampled to 2.5 m to allowe the validation of SR algorithms.


**Cadastral Data:** Cadastral data is a vector-based dataset provided by BEV and represent the legal registry for land ownership and management in Austria. It is published semi-annualy at the beginning of April and October. The data is rasterized to 2.5 m and saved as a single band image with each cadastral class reprsented by a unique integer value. In total, 26 cadastral classes are available, ranging from buildings, forest, pastures, road infrastructure to glaciers, open rock and others (see Table below).

| Code | Land Use Category                     |
|------|--------------------------------------|
| 41   | Buildings                           |
| 83   | Adjacent building areas             |
| 59   | Flowing water                       |
| 60   | Standing water                      |
| 61   | Wetlands                            |
| 64   | Waterside areas                     |
| 40   | Permanent crops or gardens          |
| 48   | Fields, meadows or pastures         |
| 57   | Overgrown areas                     |
| 55   | Krummholz                           |
| 56   | Forests                             |
| 58   | Forest roads                        |
| 42   | Car parks                           |
| 62   | Low vegetation areas                |
| 63   | Operating area                      |
| 65   | Roadside areas                      |
| 72   | Cemetery                            |
| 84   | Mining areas, dumps and landfills   |
| 87   | Rock and scree surfaces             |
| 88   | Glaciers                            |
| 92   | Rail transport areas                |
| 95   | Road traffic areas                  |
| 96   | Recreational area                   |
| 52   | Gardens                             |
| 54   | Alps                                |



Both Orthofoto and cadastral data is temporally and spatially aligned with each other and the Sentinel-2 images. A publication detailing this processing is currently under review.

**Dataset Content:**

The dataset contains several parts:

- **Sentinel-2:** Image tile (128x128, uint16) featuring all bands at 10 m to 60 m resolution.
- **Sentinel-2 harmonized:** Image tile (128x128, uint16) featuring all bands at 10 m to 60 m resolution. Each tile is histogram matched to the corresponding Orthofoto tile.
- **Orthofoto:** Image tile (512x512, uint8) featuring RGBNIR bands at 2.5 m resolution.
- **Orthofoto harmonized:** Image tile (512x512, uint16) featuring RGBNIR bands at 2.5 m resolution. Each tile is histogram matched to the corresponding Sentinel-2 tile.
- **Cadastral mask:** Image tile (512x512, uint8) single band at 2.5 m resolution. Each class is represented as its integer code.

Additional metadata available for each tortilla includes:
- Spatial position (lon / lat): latitude andlongitude in EPSG:4326
- Original Sentinel-2 tile id (s2_full_id)
- Sentinel-2 tile capture date (time)
- Sentinel-2 cloud score (cs_cdf)
- Harmonization correlation (low_corr)
- Original Orthofoto tile id (ARCHIVNR)
- Orthofoto capture time frame (ortho_begin_date and ortho_end_date)
- Percentage distribution (dist_*) of cadastral classes for each mask including NoData values (from 0-1)
- Aggregated Corine Landcover Classification (corine) sampled from a 100m CLC raster for the centroid of each tile:
	- 1: urban
	- 2: water
	- 3: agricultural
	- 4: forest
	- 5: other
	- 6: bare_rock
	- 7: glacier

All tiles have NoData values as 0 and are compressed with:

- Compression algorithm: zst  
- Compression level: 13  
- Predictor: 2  
- Interleave mode: band  

The dataset is organized in TACO multi-part files for direct use with the TACO framework. 

"""

collection_object = tacotoolbox.datamodel.Collection(
    id="sen2austria",
    title="SEN2AUSTRIA: A Super-Resolution Validation Dataset with Austrian Orthofoto Imagery and Cadstral Ground Truth Data",
    dataset_version="0.0.1",  # Update version accordingly
    description=description,
    licenses=["cc-by-4.0"],
    extent={
        "spatial": [[9.5307489063037725, 46.3724547018787021, 17.1607732062705942, 49.0205246407950739]],
        # Define spatial extent
        "temporal": [["2021-05-11T00:00:00Z", "2023-09-27T00:00:00Z"]]  # Define temporal extent
    },
    providers=[{
        "name": "a",  # Update provider name
        "roles": ["host"],
        "links": [{"href": "", "rel": "source", "type": "text/html"}],
    }],
    keywords=["remote-sensing", "super-resolution", "deep-learning", "sentinel-2", "orthofoto"],
    task="super-resolution",
    curators=[{
        "name": "Samuel Hollendonner",
        "organization": "TU Vienna, Department of Geodesy and Geoinformation",
        "email": ["samuel@plix.at"],
        "links": [{"href": "https://github.com/Zerhigh", "rel": "homepage", "type": "text/html"}],
    },
        {
            "name": "Julio Contreras",
            "organization": "Image & Signal Processing",
            "email": ["julio.contreras@uv.es"],
            "links": [{"href": "https://juliocontrerash.github.io/", "rel": "homepage", "type": "text/html"}],
        }],
    split_strategy="none",
    discuss_link={"href": "", "rel": "source", "type": "text/html"},
    raw_link={"href": "", "rel": "source", "type": "text/html"},
    optical_data={"sensor": "sentinel2msi", "sensor": "austrian_orthofoto_series"},
    labels={"label_classes": [{"name": "Buildings", "category": 41},
                              {"name": "Adjacent building areas", "category": 83},
                              {"name": "Flowing water", "category": 59},
                              {"name": "Standing water", "category": 60},
                              {"name": "Wetlands", "category": 61},
                              {"name": "Waterside areas", "category": 64},
                              {"name": "Permanent crops or gardens", "category": 40},
                              {"name": "Fields, meadows or pastures", "category": 48},
                              {"name": "Overgrown areas", "category": 57},
                              {"name": "Krummholz", "category": 55},
                              {"name": "Forests", "category": 56},
                              {"name": "Forest roads", "category": 58},
                              {"name": "Car parks", "category": 42},
                              {"name": "Low vegetation areas", "category": 62},
                              {"name": "Operating area", "category": 63},
                              {"name": "Roadside areas", "category": 65},
                              {"name": "Cemetery", "category": 72},
                              {"name": "Mining areas, dumps and landfills", "category": 84},
                              {"name": "Rock and scree surfaces", "category": 87},
                              {"name": "Glaciers", "category": 88},
                              {"name": "Rail transport areas", "category": 92},
                              {"name": "Road traffic areas", "category": 95},
                              {"name": "Recreational area", "category": 96},
                              {"name": "Gardens", "category": 52},
                              {"name": "Alps", "category": 54}],
            "label_description": "Official labels of the Austrian Cadaster, saved in high-resolution masks."},
    scientific={
        "doi": "",
        "citation": "a",  # Update citation
        "summary": "",
        "publications": [{
            "doi": "",
            "citation": "a",  # Update citation
            "summary": "",
        }]
    })

# Get the path of the tortilla file and create the directory if needed
full_path = table["tortilla_path"].iloc[0]
directory = pathlib.Path(full_path).parts[0]

t = pathlib.Path(ROOT_DIR / "austria.taco")

# Generate the final output file using the samples and collection objects
output_file = tacotoolbox.create(
    samples=samples_obj,
    collection=collection_object,
    output=t
)
