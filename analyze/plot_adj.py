#from Caulimate.Utils.Visualization import plot_sparsity_matrix
import numpy as np
import torch
import torch.nn as nn

from metrics import *


# def plot_adj_matrix(est, gt, index, plt_num=True):
#     fig = plot_sparsity_matrix(est, 'Estimated B', plt_num=plt_num)
#     fig.savefig(f'./Figures/est_{index}.pdf', format='pdf')
#     np.save(f'./B_results/est_{index}.npy', est)

#     fig = plot_sparsity_matrix(gt, 'Ground Truth B', plt_num=plt_num)
#     fig.savefig(f'./Figures/gt_{index}.pdf', format='pdf')
#     np.save(f'./B_results/gt_{index}.npy', gt)

def load_adj_matrix(index):
    est = np.load(f'./B_results/est_{index}.npy')
    gt = np.load(f'./B_results/gt_{index}.npy')
    return est, gt

def save_adj_matrix(est, gt, index):
    np.save(f'./B_results/est_{index}.npy', est)
    np.save(f'./B_results/gt_{index}.npy', gt)

lag = np.array([[[1,0,0,0],
                [1,1,0,0],
                [0,1,1,0],
                [0,0,1,1]]])
 
lag_h = np.array([[[1,0,0,0],
                    [1,1,0,0],
                    [1,0,1,0],
                    [0,0,1,1]]])

ins = np.array([[0,0,0,0],
                [1,0,0,0],
                [1,0,0,0],
                [0,0,1,0]])

ins_hat = np.array([[0,0,0,0],
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,1,0,0]])

dag = np.array([[0,1,0],
                [0,0,1],
                [0,0,0]]) 

dag_hat = np.array([[0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,0],
                    [1,1,0,0]]) 

dag = np.array([[0,1,0,0],
                [0,0,1,0],
                [0,0,0,0],
                [1,1,0,0]]) 

dag_hat = np.array([[0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,0],
                    [1,1,0,0]]) 

index = f'x_dim_{ins.shape[-1]}_z_dim_{ins.shape[-1]}'

# plot_adj_matrix(est, gt, index, False)
save_adj_matrix(dag_hat, dag, index)

results = evaluate_structure(lag, ins, dag, lag_h, ins_hat, dag_hat)
print(results)