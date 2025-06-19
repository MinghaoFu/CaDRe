# wind_vector.py
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
from Caulimate.Utils.Visualization import plot_causal_graph, quick_map
from Caulimate.Utils.Tools import recover_coordinates
import os

DATASET_DIR = os.environ.get('DATASET_DIR')

u = xr.open_mfdataset(os.path.join(DATASET_DIR, 'WeatherBench_data_full/u_component_of_wind/*.nc'), combine='by_coords')
v = xr.open_mfdataset(os.path.join(DATASET_DIR, 'WeatherBench_data_full/v_component_of_wind/*.nc'), combine='by_coords')

u = recover_coordinates(u)
v = recover_coordinates(v)

for idx in range(0, 1000, 100):

    v_slice = v.isel(time=idx).sel(level=850)['v']
    u_slice = u.isel(time=idx).sel(level=850)['u']

    lon = u_slice.coords['lon']
    lat = u_slice.coords['lat']
    center=180
    # Plotting
    plt.figure(figsize=(14, 6))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=center)})
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE,linewidth=0.3)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':',linewidth=0.3)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False  # Disable labels at the top
    gl.right_labels = False  # Disable labels on the right
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    ax.set_extent([120, 300, -30, 30], crs=ccrs.PlateCarree())
    # # Add coordinate ticks
    # ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())


    plt.quiver(lon + 180, lat, u_slice, v_slice, scale=500, color='red', width=0.001)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'./Figures/PacificWind_{idx}.pdf', format='pdf', dpi=300, bbox_inches='tight')