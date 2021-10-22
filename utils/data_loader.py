import os

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import torch
import torch_geometric.transforms as T

from .get_graph_partitions import get_graph_partitions, get_train_node_per_part, neighbor_approx
from .get_label_partitions import get_label_partitions

def load_dataset(dataset_name, num_parts, 
                 partition_method=0):
    # load OGB dataset
    dataset = PygNodePropPredDataset(name      = dataset_name,
                                     transform = T.ToSparseTensor())
    data      = dataset[0]
    split_idx = dataset.get_idx_split()
    
    # Create local partitions
    if partition_method == 0: # use metis based on graph structure
        print('Partition with metis based on graph structure')
        parts = get_graph_partitions(dataset      = dataset_name, 
                                     num_clusters = num_parts, 
                                     data = data, 
                                     root = os.getcwd())
    elif partition_method == 1: # use label heterogeneous 
        print('Partition based on label heterogeneous')
        parts = get_label_partitions(dataset      = dataset_name, 
                                     num_clusters = num_parts, 
                                     data = data, 
                                     root = os.getcwd())
    elif partition_method == 2: # use metis based on graph structure + overhead
        print('Partition with metis based on graph structure + overhead')
        parts = get_graph_partitions(dataset      = dataset_name, 
                                     num_clusters = num_parts, 
                                     data = data, 
                                     root = os.getcwd())
        
        # convert PyG data structure to scipy's sparse_coo for metis partition
        import numpy as np
        from scipy.sparse import csr_matrix
        
        row = data.adj_t.storage.row().numpy()
        col = data.adj_t.storage.col().numpy()
        num_nodes = data.adj_t.size(0)

        all_nodes = np.arange(num_nodes)
        sparse_coo_adj = csr_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))
        
        parts = [neighbor_approx(sparse_coo_adj, part) for part in parts]
    else:
        print('Unknown partition method')
    
    train_parts = get_train_node_per_part(num_nodes   = data.x.size(0),
                                          train_nodes = split_idx['train'].numpy(), 
                                          parts       = parts)

    data.parts = [torch.tensor(part) for part in parts]
    data.train_parts = [torch.tensor(part) for part in train_parts]
    
    # add eval
    evaluator = Evaluator(name=dataset_name)
    
    return dataset, data, split_idx, evaluator

def sparse_tensor_to_edge_indices(adj):
    row = adj.storage.row()
    col = adj.storage.col() 
    return torch.stack([row, col])