import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import numpy as np

import os
import seaborn as sns
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from Caulimate.Data.WB import create_adjacency_matrix

data_dir = os.getenv('DATASET_DIR')
SST_DATA_PATH = os.path.join(data_dir, 'WeatherBench_data_full/temperature_850/*.nc')
z500 = xr.open_mfdataset(SST_DATA_PATH, combine='by_coords')
z500.shape

# adj_mat = create_adjacency_matrix(u_slice.values, v_slice.values, lon, lat)
# fig, ax = plt.subplots(
#     figsize=(10, 5),
#     subplot_kw={'projection': ccrs.PlateCarree()},
# )

# offset = 2
# ax.set_extent([np.min(coords[:, 1]) - offset, np.max(coords[:, 1]) + offset, np.min(coords[:, 0]) - offset, np.max(coords[:, 0]) + offset], crs=ccrs.PlateCarree())
# #ax.stock_img()
# ax.add_feature(cartopy.feature.LAND)
# ax.add_feature(cartopy.feature.OCEAN)
# ax.add_feature(cartopy.feature.COASTLINE,linewidth=0.3)
# ax.add_feature(cartopy.feature.BORDERS, linestyle=':',linewidth=0.3)
# ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
# ax.add_feature(cartopy.feature.RIVERS)


# mat = adj_mat + _dag
# mat[np.abs(mat) <= 0.4] = 0  
# plot_causal_graph(coords, mat, ax)
# plt.show()
# mat.nonzero()


