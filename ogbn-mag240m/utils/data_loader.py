import time
import os.path as osp

import numpy as np

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor

from pytorch_lightning import LightningDataModule

from torch_sparse import SparseTensor
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset

from .get_smaller_graph import print_dataset_stat, get_part_adjs, get_part_feats, get_part_nodes_split

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )
    
class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 force_regen: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.force_regen = force_regen
        
    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        print_dataset_stat(dataset)
        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path) or self.force_regen:
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True) 
            edge_index = get_part_adjs(dataset) 
            num_nodes = np.max(edge_index)+1
            
            edge_index = torch.from_numpy(edge_index)
            
            adj_t = SparseTensor(
                row=edge_index[0, :], 
                col=edge_index[1, :],
                sparse_sizes=(num_nodes, num_nodes))
            
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.x = torch.from_numpy(get_part_feats(dataset)).share_memory_()
        
        activate_node_labels, node_split = get_part_nodes_split(dataset)
        self.train_idx = torch.from_numpy(node_split['train']).long().share_memory_()
        self.val_idx   = torch.from_numpy(node_split['valid']).long().share_memory_()
        self.test_idx  = torch.from_numpy(node_split['test-dev']).long().share_memory_()
        self.y = torch.from_numpy(activate_node_labels)
        
        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        self.adj_t = torch.load(path).long()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        x = self.x[n_id].to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])