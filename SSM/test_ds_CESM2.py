import sys
sys.path.append('..')

from LiLY.modules.tv_golem import GolemModel
import torch
import torch.nn as nn
import torch.optim as optim
import os, pwd

import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ipdb
import pytorch_lightning as fpl
import wandb
from pytorch_lightning.loggers import WandbLogger
from einops import repeat
from LiLY.modules.CESM2 import CESM2ModularShiftsFixedB
import cartopy.crs as ccrs
import cartopy
import numpy as np
import matplotlib.pyplot as plt 

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from tqdm import tqdm
from einops import repeat

from Caulimate.Data.SimLinGau import LinGauSuff
from Caulimate.Data.SimDAG import simulate_random_dag, simulate_weight, simulate_time_vary_weight
from Caulimate.Utils.Visualization import save_DAG, make_dots, plot_causal_graph, quick_map
from Caulimate.Utils.Tools import check_tensor, check_array, load_yaml, makedir, lin_reg_init, dict_to_class, save_log, bin_mat, center_and_norm, get_free_gpu
from Caulimate.Utils.GraphMetric import count_graph_accuracy
from Caulimate.Data.CESM2.dataset import CESM2_grouped_dataset, downscale_dataset
from Caulimate.Utils.GraphUtils import eudistance_mask, decycle_till_dag

DATASET_DIR = os.environ.get('DATASET_DIR')
MODEL_DIR = os.environ.get('MODEL_DIR')

DATA_DIR = os.path.join(DATASET_DIR, 'CESM2')
DOWNSCALE_PATH = os.path.join(DATA_DIR, 'downscaled_pacific_CESM2.txt')
DOWNSCALE_METADATA_PATH = os.path.join(DATA_DIR, 'downscaled_metadata.pkl')

SAVE_DIR = os.path.join(MODEL_DIR, 'ClimateModel/LinGau/CESM2')
makedir(SAVE_DIR)
save_test_dir = './downscale_CESM2_eud_mask'

CKPT_PATH="/fsx/homes/Minghao.Fu@mbzuai.ac.ae/workspace/climate-project/SSM/climate/5nikihdb/checkpoints/epoch=1044-step=98230.ckpt"

# if torch.cuda.is_available():   
#     os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()
#     print(f"--- Selected GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}")


def randomly_zero_elements(adj_matrix, mask, keep=True, zero_fraction=0.05, one_fraction=0.0025, seed=1):
    if seed is not None:
        np.random.seed(seed)

    nonzero_indices = np.argwhere(adj_matrix != 0)
    num_to_zero = int(len(nonzero_indices) * zero_fraction)
    set_zero_indices = nonzero_indices[np.random.choice(len(nonzero_indices), num_to_zero, replace=False)]

    zero_indices = np.argwhere(mask == 1)
    num_to_one = int(len(zero_indices) * one_fraction)
    set_one_indices = zero_indices[np.random.choice(len(zero_indices), num_to_one, replace=False)]
    #set_one_indices = np.array([index for index in set_one_indices if mask[index[0], index[1]] != 0])
    for idx in set_zero_indices:
        adj_matrix[tuple(idx)] = 1
    for idx in set_one_indices:
        adj_matrix[tuple(idx)] = 1
    return adj_matrix, set_zero_indices.tolist(), set_one_indices.tolist()

args = {
    'data_path': "/l/users/minghao.fu/dataset/CESM2/CESM2_pacific_grouped_SST.nc",
    'noise_type': 'gaussian_ev',
    'load_data': True,
    'graph_type': 'ER',
    'num': 6000,
    'scale': 0.5,
    'pi': 10,
    'd_X': None,
    'degree': 4,
    'cos_len': 1000,
    'max_eud': 40,
    'equal_variances': True,

    'train': True,
    'pretrain': False,
    'checkpoint_path': None,
    'regression_init': False,
    'loss': {
        'likelihood': 1.0,
        'L1': 1.e-2,
        'dag': 1.e-2
    },
    'reg_thres': 0.05,
    'ddp': False,
    'pre_epoch': 0,
    'epoch': 10000,
    'init_epoch': 100,
    'batch_size': 64,
    'lag': 10,
    'synthetic': False,
    'time_varying': False,
    'sparse': False,

    'seed': 2,
    'gt_init': False,
    'embedding_dim': 5,
    'spectral_norm': False,
    'tol': 0.0,
    'graph_thres': 0.3,
    'DAG': 0.8,
    'save_dir': "/l/users/minghao.fu/logs/ClimateModel/LinGau/CESM2",

    'condition': "ignavier",
    'decay_type': "step",
    'optimizer': "ADAM",
    'weight_decay': 0.0,
    'lr': 1.e-4,
    'gradient_noise': None,
    'step_size': 1000,
    'gamma': 0.5,
    'decay': [200, 400, 800, 1000],
    'betas': [0.9, 0.999],
    'epsilon': 1.e-8,
    'momentum': 0.9
}

# model = CESM2ModularShiftsFixedB.load_from_checkpoint(checkpoint_path=CKPT_PATH, strict=False)
# model.eval()

# dataset = downscale_dataset(path=DOWNSCALE_PATH, metadata_path=DOWNSCALE_METADATA_PATH)#CESM2_grouped_dataset(args.data_path, num_area=1)[0]
mask = np.load(os.path.join(save_test_dir, 'mask.npy'))
# Bs = check_array(model.generate_Bs(check_tensor(dataset.T).squeeze(1))) * mask
# np.save(os.path.join(save_test_dir, 'Bs.npy'), Bs)

coords = np.load(os.path.join(save_test_dir, 'coords.npy'))
Bs = np.load(os.path.join(save_test_dir, 'Bs.npy'))
time_ind = 1000
Bt = Bs[time_ind]

Bt[np.abs(Bt) < 0.755] = 0 
Bt = decycle_till_dag(Bt)
# physical prior based on climate
one_list = [[67, 52], [79, 51], [79, 65],[47, 46],[46, 45],[45, 44],[44, 43],[43, 42],[33,32],[32,31], [54, 40],[40,26],[50,48],[50,49],[68,54]]
one_list = one_list
for pair in one_list:
    Bt[pair[0], pair[1]] = 1
    Bt[pair[1], pair[0]] = 0

zero_list = [[17,59],[33,73],[6,46],[17,59],[59,17],[6,46],[6,47],[6,48],[23,73],[71,31], [78,67], [64,53]]
for pair in zero_list:
    Bt[pair[0], pair[1]] = 0

# corrupt_list = [[77,48],[29,70],[0,40],[69,28],[53,37],[76,49],[21,60],[6,25],[9,35],[21,47],[77,61],[68,41],[51,22]]
# corrupt_list = corrupt_list[:-5]
# for pair in corrupt_list:
#     Bt[pair[0], pair[1]] = 1

Bt, dis_list, app_list = randomly_zero_elements(Bt, mask=mask)

print(f'Averaged Degree: {np.mean(np.sum(bin_mat(Bt), axis=1))}')  
center=180
# Assuming coords is a list of tuples, where each tuple is (lat, lon)
adjusted_coords = [(lat, 180 - lon if lon <= 180 else lon - 180) for lat, lon in coords]

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

def offset_path(start, end, offset):
    start_lon, start_lat = start
    end_lon, end_lat = end
    mid_lon = (start_lon + end_lon) / 2 + offset
    mid_lat = (start_lat + end_lat) / 2 + offset
    return [(start_lon, start_lat), (mid_lon, mid_lat), (end_lon, end_lat)]

# 根据邻接矩阵绘制因果连接
offset = 0.5  # 偏移量

for i in range(len(Bt)):
    for j in range(len(Bt)):
        if Bt[i, j] != 0:
            
            point1 = coords[i]
            point2 = coords[j]
            # ax.plot(*point1, 'ro', transform=ccrs.PlateCarree())  
            # ax.plot(*point2, 'bo', transform=ccrs.PlateCarree())  
            # ax.annotate(f'i', xy=point1, color='green', transform=ccrs.PlateCarree())  # annotate point1
            # ax.annotate(f'j', xy=point2, color='green', transform=ccrs.PlateCarree())  # annotate point2
            # 处理跨越日期变更线的情况
            # if abs(point1[1] - point2[1]) > 180:
            #     if point1[1] > point2[1]:
            #         point2 = (point2[0], point2[1] + 360)
            #     else:
            #         point1 = (point1[0], point1[1] + 360)
            path = offset_path(point1, point2, offset * (i + j))
            xs, ys = zip(*path)
            ax.plot(xs, ys, color='red', linewidth=1, transform=ccrs.PlateCarree())
            color = 'black'
            if [i, j] in dis_list:  
                color = 'red'
            elif [i, j] in app_list:
                color = 'blue'
            ax.annotate('', xy=(point2[1], point2[0]), xytext=(point1[1], point1[0]),
                        arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='->', lw=1, connectionstyle="arc3"), #arc3,rad=.2
                        transform=ccrs.PlateCarree())
            

for idx, (lat, lon) in enumerate(coords):
    ax.annotate(f'{idx}', xy=(lon - 180, lat), xytext=(lon - 180, lat),
                            arrowprops=dict(facecolor='red', arrowstyle='->', lw=1))
    
ax.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())
name = 'c_t'
makedir(os.path.join(save_test_dir, name))
plt.savefig(os.path.join(save_test_dir, name, f'causal_graph_{time_ind}.pdf'), format='pdf', bbox_inches='tight')
np.save(os.path.join(save_test_dir, name, f'causal_graph_{time_ind}.npy'), Bt)
np.save(os.path.join(save_test_dir, name, f'dis_list_{time_ind}.npy'), np.array(dis_list))
np.save(os.path.join(save_test_dir, name, f'app_list_{time_ind}.npy'), np.array(app_list))