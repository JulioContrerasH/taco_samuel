import numpy as np

from utils_histogram import *
import pathlib
import time
import os
from multiprocessing import Pool
import rasterio as rio
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage.exposure import match_histograms

ROOT_DIR = pathlib.Path("/data/USERS/shollend/taco")
df = pd.read_csv(ROOT_DIR / "metadata_updated.csv")

correlation = df['low_corr'].corr(df['cs_cdf'])
print(correlation)

low_ = df[df['low_corr'] < 0.3]
low_.to_csv('low_corr_less_0dot3.csv', index=False)

plot = False
if plot:
    # Plot histogram for column 'A'
    df['low_corr'].plot(kind='hist', bins=10, edgecolor='black', alpha=0.7)
    # Customize plot
    plt.title('Histogram of Column A')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()


