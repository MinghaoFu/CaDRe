import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import math
import pandas as pd
import xarray as xr
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from Caulimate.Data.CESM2.dataset import CESM2_grouped_dataset, downscale_dataset
from Caulimate.Utils.Visualization import quick_map
from Caulimate.Utils.Metrics import MAE, MSE, MAPE, RMSE, R_squared
from Caulimate.Utils.Tools import makedir, check_tensor

DATA_DIR = os.path.join(os.getenv('DATASET_DIR'), 'CESM2') # you could modify it to your path
DOWNSCALE_PATH = os.path.join(DATA_DIR, 'downscaled_pacific_CESM2.txt')
DOWNSCALE_METADATA_PATH = os.path.join(DATA_DIR, 'downscaled_metadata.pkl')

NUM_AREA = 1
TIME_IDX = 1000
SAVE_DIR = './Figures'

data_array = np.loadtxt(DOWNSCALE_PATH)


# Define the path to save the NumPy array
npy_save_path = "./CESM2_pacific_grouped_SST.npy"

# Save the NumPy array to a .npy file
np.save(npy_save_path, data_array)

print(f"Data saved to {npy_save_path}")
