from itertools import permutations

import numpy as np

import numpy as np

def generate_random_dag(n_nodes, edge_prob=0.3, seed=None):
    """
    Generate a random DAG represented as an adjacency matrix.

    Parameters:
    - n_nodes (int): Number of nodes in the DAG.
    - edge_prob (float): Probability of creating an edge between nodes (0 to 1).
    - seed (int, optional): Seed for random number generation.

    Returns:
    - adj_matrix (np.ndarray): Adjacency matrix of the DAG with shape (n_nodes, n_nodes).
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Create an upper triangular matrix to ensure no cycles
    adj_matrix = np.triu((np.random.rand(n_nodes, n_nodes) < edge_prob).astype(int), k=1)
    perm = np.random.permutation(adj_matrix.shape[0])

    # Apply the permutation to both rows and columns
    permuted_adj_matrix = adj_matrix[perm, :][:, perm]
    
    return permuted_adj_matrix

    return adj_matrix



def count_precision_recall_f1(tp, fp, fn):
    # Precision
    if tp + fp == 0:
        precision = None
    else:
        precision = float(tp) / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = None
    else:
        recall = float(tp) / (tp + fn)

    # F1 score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def count_dag_accuracy(dag_true, dag_est):
    d = dag_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(dag_est)
    cond = np.flatnonzero(dag_true)
    cond_reversed = np.flatnonzero(dag_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    pred_size = len(pred)
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(dag_est + dag_est.T))
    cond_lower = np.flatnonzero(np.tril(dag_true + dag_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(reverse) + len(false_pos),
                                                      fn=len(false_neg))
    return {'shd': shd, 'pred_size': pred_size, 'precision': precision,
            'recall': recall, 'f1': f1}


def count_und_accuracy(und_true, und_est):
    d = len(und_true)
    und_triu_true = und_true[np.triu_indices(d, k=1)]
    und_triu_est = und_est[np.triu_indices(d, k=1)]
    pred = np.flatnonzero(und_triu_est)
    cond = np.flatnonzero(und_triu_true)
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    # compute ratio
    nnz = len(pred)
    cond_neg_size = len(und_triu_true) - len(cond)
    fdr = float(len(false_pos)) / max(nnz, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(false_pos)) / max(cond_neg_size, 1)
    try:
        f1 = len(true_pos) / (len(true_pos) + 0.5 * (len(false_pos) + len(false_neg)))
    except:
        f1 = None
    # structural hamming distance
    extra_lower = np.setdiff1d(pred, cond, assume_unique=True)
    missing_lower = np.setdiff1d(cond, pred, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower)
    return {'f1': f1, 'precision': 1 - fdr, 'recall': tpr,
            'shd': shd, 'pred_size': nnz}


def standardize_num_latents(Z_lagged_dag_true, Z_instan_dag_true,
                            Z_lagged_dag_est, Z_instan_dag_est):
    diff = len(Z_instan_dag_true) - len(Z_instan_dag_est)
    # If number of latent variables are different, pad them
    if diff == 0:
        pass
    elif diff < 0:
        pad_num = abs(diff)
        Z_lagged_dag_true, Z_instan_dag_true \
            = pad_latents(Z_lagged_dag_true, Z_instan_dag_true, pad_num)
    else:
        pad_num = abs(diff)
        Z_lagged_dag_est, Z_instan_dag_est \
            = pad_latents(Z_lagged_dag_est, Z_instan_dag_est, pad_num)
    return Z_lagged_dag_true, Z_instan_dag_true, Z_lagged_dag_est, Z_instan_dag_est


def pad_latents(Z_lagged_dag, Z_instan_dag, pad_num=1):
    assert pad_num >= 0
    num_latents = len(Z_instan_dag)
    num_lags = len(Z_lagged_dag)
    # Pad Z_instan_dag
    Z_instan_dag_padded = Z_instan_dag
    Z_instan_dag_padded = np.hstack([Z_instan_dag_padded, np.zeros((num_latents, pad_num))])
    Z_instan_dag_padded = np.vstack([Z_instan_dag_padded, np.zeros((pad_num, num_latents + pad_num))])
    # Pad Z_lagged_dag
    Z_lagged_dag_padded = Z_lagged_dag
    Z_lagged_dag_padded = np.dstack([Z_lagged_dag_padded, np.zeros((num_lags, num_latents, pad_num))])
    Z_lagged_dag_padded = np.hstack([Z_lagged_dag_padded, np.zeros((num_lags, pad_num, num_latents + pad_num))])
    return Z_lagged_dag_padded, Z_instan_dag_padded


def find_best_permutation(Z_lagged_dag_true, Z_instan_moral_true,
                          Z_lagged_dag_est, Z_instan_moral_est):
    num_latents = len(Z_instan_moral_true)
    num_lags = len(Z_lagged_dag_true)
    indices = list(range(num_latents))
    best_shd = float('inf')
    best_P = None
    for permuted_indices in permutations(indices):
        P = np.zeros_like(Z_instan_moral_true)
        P[range(num_latents), permuted_indices] = 1
        permuted_Z_instan_moral_est = P.T @ Z_instan_moral_est @ P
        P_broadcast = P[np.newaxis, :, :]
        P_broadcast_T = P.T[np.newaxis, :, :]
        permuted_Z_lagged_dag_est = P_broadcast_T @ Z_lagged_dag_est @ P_broadcast
        # Calculate SHD of Z_instan_moral
        shd_instan = count_und_accuracy(Z_instan_moral_true, permuted_Z_instan_moral_est)['shd']
        # Calculate SHD of Z_lagged
        extended_permuted_Z_lagged_dag_est = np.zeros(((num_lags + 1) * num_latents, (num_lags + 1) * num_latents))
        extended_permuted_Z_lagged_dag_est[num_latents:, :num_latents] = permuted_Z_lagged_dag_est.reshape(num_lags * num_latents, num_latents)
        extended_Z_lagged_dag_true = np.zeros(((num_lags + 1) * num_latents, (num_lags + 1) * num_latents))
        extended_Z_lagged_dag_true[num_latents:, :num_latents] = Z_lagged_dag_true.reshape(num_lags * num_latents, num_latents)
        shd_lagged = count_dag_accuracy(extended_Z_lagged_dag_true, extended_permuted_Z_lagged_dag_est)['shd']
        shd = shd_instan + shd_lagged
        if shd < best_shd:
            best_shd = shd
            best_P = P
    return best_P


def get_moral_graph(dag):
    """Compute moral graph.
    Each column represents the parents of each variable."""
    if not ((dag == 0) | (dag == 1)).all():
        raise ValueError('dag should take value in {0,1}')
    num_vars = len(dag)
    I = np.eye(num_vars)
    moral_graph = (I + dag) @ (I + dag).T
    moral_graph = (moral_graph != 0).astype(int)
    moral_graph[np.arange(num_vars), np.arange(num_vars)] = 0
    return moral_graph


def evaluate_structure(Z_lagged_dag_true, Z_instan_dag_true, X_dag_true,
                       Z_lagged_dag_est, Z_instan_dag_est, X_dag_est):
    """
    X_dag_true and X_dag_true are binary matrices of shape (num_measured, num_measured)
    - X_dag[i, j] = 1 if X_i -> X_j; otherwise the value is 0

    Z_instan_dag_true and Z_instan_dag_est are binary matrices of shape (num_latent, num_latent)
    - Z_instan_dag[i, j] = 1 if Z_i -> Z_j; otherwise the value is 0

    Z_lagged_dag_true and Z_lagged_dag_est are binary matrices of shape (num_lags, num_latent, num_latent)
    - Z_lagged_dag[k, i, j] = 1 if Z_i at time t - 1 - k points to Z_j at time t; otherwise the value is 0
    """
    if not ((Z_lagged_dag_true == 0) | (Z_lagged_dag_true == 1)).all():
        raise ValueError('Z_lagged_adj_true should take value in {0,1}')
    if not ((Z_instan_dag_true == 0) | (Z_instan_dag_true == 1)).all():
        raise ValueError('Z_instan_adj_true should take value in {0,1}')
    if not ((X_dag_true == 0) | (X_dag_true == 1)).all():
        raise ValueError('X_adj_true should take value in {0,1}')
    if not ((Z_lagged_dag_est == 0) | (Z_lagged_dag_est == 1)).all():
        raise ValueError('Z_lagged_adj_est should take value in {0,1}')
    if not ((Z_instan_dag_est == 0) | (Z_instan_dag_est == 1)).all():
        raise ValueError('Z_instan_adj_est should take value in {0,1}')
    if not ((X_dag_est == 0) | (X_dag_est == 1)).all():
        raise ValueError('X_adj_est should take value in {0,1}')
    # Standardize number of latent variables
    Z_lagged_dag_true, Z_instan_dag_true, Z_lagged_dag_est, Z_instan_dag_est \
        = standardize_num_latents(Z_lagged_dag_true, Z_instan_dag_true,
                                  Z_lagged_dag_est, Z_instan_dag_est)
    result = dict()
    # Calculate accuracy of edges between X
    result['X_dag'] = count_dag_accuracy(X_dag_true, X_dag_est)
    # Convert Z_instan_dag to moral graph
    Z_instan_moral_true = get_moral_graph(Z_instan_dag_true)
    Z_instan_moral_est = get_moral_graph(Z_instan_dag_est)
    # Find best permutation that minimizes the SHDof Z_instan_moral and Z_lagged_dag
    P = find_best_permutation(Z_lagged_dag_true, Z_instan_moral_true,
                              Z_lagged_dag_est, Z_instan_moral_est)
    best_Z_lagged_dag_est = P.T @ Z_lagged_dag_est @ P
    best_Z_instan_moral_est = P.T @ Z_instan_moral_est @ P
    # Calculate accuracy of instantaneous edges (of moral graph) between Z
    result['Z_instan_moral_graph'] = count_und_accuracy(Z_instan_moral_true, best_Z_instan_moral_est)
    # Calculate accuracy of time-lagged edges between Z
    num_latents = len(Z_instan_moral_true)
    num_lags = len(Z_lagged_dag_true)    # Shape: (num_lags, num_latents, num_latents)
    # We first extend the time-lagged graph in order to reuse count_dag_accuracy
    extended_best_Z_lagged_dag_est = np.zeros(((num_lags + 1) * num_latents, (num_lags + 1) * num_latents))
    extended_best_Z_lagged_dag_est[num_latents:, :num_latents] = best_Z_lagged_dag_est.reshape(num_lags * num_latents, num_latents)
    extended_Z_lagged_dag_true = np.zeros(((num_lags + 1) * num_latents, (num_lags + 1) * num_latents))
    extended_Z_lagged_dag_true[num_latents:, :num_latents] = Z_lagged_dag_true.reshape(num_lags * num_latents, num_latents)
    result['Z_lagged_dag'] = count_dag_accuracy(extended_Z_lagged_dag_true, extended_best_Z_lagged_dag_est)
    return result
