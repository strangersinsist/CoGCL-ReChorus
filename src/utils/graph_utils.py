import torch
import numpy as np
import scipy.sparse as sp

def build_adj_matrix(user_ids, item_ids, n_users, n_items):
    """
    Builds the adjacency matrix for a bipartite graph of users and items.
    """
    adj = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj = adj.tolil()
    
    # User-item interactions
    user_np = np.array(user_ids)
    item_np = np.array(item_ids)
    ratings = np.ones_like(user_np, dtype=np.float32)
    
    tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_users, n_items))
    adj[:n_users, n_users:] = tmp_adj
    adj[n_users:, :n_users] = tmp_adj.T
    
    adj = adj.todok()
    return adj

def gcn_norm(adj_mat, num_nodes=None, self_loop=False):
    """
    Performs symmetric normalization on the adjacency matrix.
    D^{-1/2} * A * D^{-1/2}
    """
    # Add self-loops
    if self_loop:
        adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
    
    adj_mat = adj_mat.tocoo().astype(np.float32)
    if num_nodes is None:
        num_nodes = adj_mat.shape[0]
    adj_mat = adj_mat.tocoo()
    rowsum = np.array(adj_mat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    # Convert to torch sparse tensor
    row = torch.from_numpy(norm_adj_mat.row).to(torch.long)
    col = torch.from_numpy(norm_adj_mat.col).to(torch.long)
    edge_index = torch.stack([row, col])
    edge_weight = torch.from_numpy(norm_adj_mat.data).to(torch.float32)
    
    return edge_index, edge_weight
