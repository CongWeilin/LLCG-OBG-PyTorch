
import pickle
import os
import time
from scipy.sparse import csr_matrix
import numpy as np

##############################################
##############################################
##############################################
def get_graph_partitions(dataset, num_clusters, data, root):

    save_parts_path = os.path.join(root, 'dataset/%s_%d_clusters.pkl'%(dataset, num_clusters))

    if os.path.exists(save_parts_path):
        parts = pickle.load(open(save_parts_path, 'rb'))
    else:
        # convert PyG data structure to scipy's sparse_coo for metis partition
        row = data.adj_t.storage.row().numpy()
        col = data.adj_t.storage.col().numpy()
        num_nodes = data.adj_t.size(0)

        all_nodes = np.arange(num_nodes)
        sparse_coo_adj = csr_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))

        parts = _partition_graph(sparse_coo_adj, all_nodes, num_clusters)

        with open(save_parts_path, 'wb') as f:
            pickle.dump(parts, f)
            
    return parts
##############################################
##############################################
##############################################
def get_train_node_per_part(num_nodes, train_nodes, parts):
    is_train_node = np.zeros(num_nodes)
    is_train_node[train_nodes] = 1
    
    train_parts = []
    for part in parts:
        train_part = np.where(is_train_node[part]==1)[0]
        train_parts.append(train_part)
        
    return train_parts
##############################################
##############################################
##############################################
def _partition_graph(adj, idx_nodes, num_clusters):
    os.environ['METIS_DLL'] = '/home/weilin/Downloads/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.so'
    import metis
    
    """partition a graph by METIS."""

    start_time = time.time()
    num_nodes = len(idx_nodes)

    train_adj = adj[idx_nodes, :][:, idx_nodes]
    train_adj_lil = train_adj.tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        train_ord_map[idx_nodes[i]] = i

    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes

    parts = [[] for _ in range(num_clusters)]
    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)
        

    part_size = [len(part) for part in parts]
    print('Partitioning done. %f seconds.'%(time.time() - start_time))
    print('Max part size %d, min part size %d'%(max(part_size), min(part_size)))

    return parts

def neighbor_approx(adj, idx_nodes, sample_ratio=0.1):
    adj_part = adj[idx_nodes, :]
    part_neighbors_prob = np.sum(adj_part, axis=0)
    part_neighbors_prob = part_neighbors_prob/np.sum(part_neighbors_prob)
    
    one_hop_neighbors = np.sum(part_neighbors_prob>0)
    sample_node_size = int(len(part_neighbors_prob)*sample_ratio)
    
    if sample_node_size < one_hop_neighbors:
        overhead = np.random.choice(len(part_neighbors_prob), p=part_neighbors_prob, replace=False)
    else:
        overhead = one_hop_neighbors
        
    new_idx_nodes = np.sort(np.concatenate([idx_nodes, overhead]))
    return overhead