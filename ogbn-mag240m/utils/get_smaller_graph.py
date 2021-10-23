import os
import numpy as np

from tqdm import tqdm

def print_dataset_stat(dataset, ROOT='./dataset'):
    print('number of paper nodes: ', dataset.num_papers) # number of paper nodes
    print('number of author nodes: ', dataset.num_authors) # number of author nodes
    print('number of institution nodes: ', dataset.num_institutions) # number of institution nodes
    print('dimensionality of paper features: ', dataset.num_paper_features) # dimensionality of paper features
    print('number of subject area classes: ', dataset.num_classes) # number of subject area classes

def get_part_adjs(dataset, ROOT='./dataset'):
    # get all labeled nodes
    split_dict = dataset.get_idx_split()
    all_idx = np.concatenate([np.array(arr) for arr in split_dict.values()])
    all_idx = set(all_idx)

    # create new edge index
    edge_index_path = os.path.join(ROOT, 'edge_index_label_nodes.npy')
    node_index_activate_path = os.path.join(ROOT, 'node_index_activate.npy')

    if os.path.exists(edge_index_path) and os.path.exists(node_index_activate_path):
        edge_index = np.load(open(edge_index_path, 'rb'))
        node_index_activate = np.load(open(node_index_activate_path, 'rb'))
    else:
        edge_index = dataset.edge_index('paper', 'cites', 'paper')
        write_to = 0
        for ptr in tqdm(range(edge_index.shape[1])):
            e0, e1 = edge_index[0, ptr], edge_index[1, ptr]
            if e0 in all_idx or e1 in all_idx:
                edge_index[:, write_to] = edge_index[:, ptr]
                write_to += 1
            else:
                continue
        edge_index = edge_index[:, : write_to]
        node_index_activate = np.unique(edge_index)
        np.save(node_index_activate_path, node_index_activate)
        
        # reorder nodes
        node_index = np.zeros(dataset.num_papers)
        node_index[node_index_activate] = np.arange(len(node_index_activate))

        edge_index[0, :] = node_index[edge_index[0, :]]
        edge_index[1, :] = node_index[edge_index[1, :]]

        # make symmetric
        edge_index_set = set()
        for i in tqdm(range(edge_index.shape[1])):
            e0, e1 = edge_index[0, i], edge_index[1, i]
            if e0 > e1:
                edge_index_set.add((e1, e0))
            elif e0 < e1:
                edge_index_set.add((e0, e1))

        edge_index = np.array(list(edge_index_set))
        edge_index = np.concatenate([edge_index[:, [1,0]], edge_index])
        edge_index = np.transpose(edge_index)
        
        np.save(edge_index_path, edge_index)

    return edge_index
    
def get_part_feats(dataset, ROOT='./dataset'):
    node_feats_path = os.path.join(ROOT, 'node_feats_label_nodes.npy')
    
    node_index_activate_path = os.path.join(ROOT, 'node_index_activate.npy')
    node_index_activate = np.load(open(node_index_activate_path, 'rb'))
    
    if os.path.exists(node_feats_path):
        node_feats = np.load(open(node_feats_path, 'rb'))
    else:
        node_feats = np.zeros((len(node_index_activate), 768), dtype=np.float16)
        for i in tqdm(range(len(node_index_activate))):
            node_feats[i, :] = np.array(dataset.paper_feat[node_index_activate[i]])
        np.save(node_feats_path, node_feats)
    return node_feats

def get_part_nodes_split(dataset, ROOT='./dataset'):
    node_index_activate_path = os.path.join(ROOT, 'node_index_activate.npy')
    node_index_activate = np.load(open(node_index_activate_path, 'rb'))
    
    node_labels = np.array(dataset.all_paper_label, dtype=np.int32)
    node_labels = np.nan_to_num(node_labels, nan=-1)
    
    activate_node_labels = node_labels[node_index_activate]
    
    train_val_test_split = np.zeros(dataset.num_papers)
    train_val_test_split[node_index_activate] = np.arange(len(node_index_activate))
    
    new_split_dict = {}
    split_dict = dataset.get_idx_split()
    for split in split_dict.keys():
        idx = dataset.get_idx_split(split)
        new_split_dict[split] = train_val_test_split[idx]
    
    return activate_node_labels, new_split_dict