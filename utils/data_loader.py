import os

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import torch
import torch_geometric.transforms as T

from .get_graph_partitions import get_graph_partitions, get_train_node_per_part, _neighbor_approx
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
        parts = [neighbor_approx(part) for part in parts]
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