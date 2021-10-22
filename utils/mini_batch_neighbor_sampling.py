import torch

from typing import List, Optional

from torch_sparse import SparseTensor
from torch_geometric.loader.neighbor_sampler import EdgeIndex, Adj

class DataIterator(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.iterator = iter(self.sampler)
    
    def sample(self, ):
        try:
            batch_size, n_id, adjs = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.sampler)
            batch_size, n_id, adjs = next(self.iterator)
        return batch_size, n_id, adjs
    
class NeighborSamplerSharedGraphDiffIndex(object):
    
    def __init__(self, edge_index: torch.Tensor, 
                 sizes: List[int],
                 node_idx: List[torch.Tensor],
                 num_nodes: Optional[int] = None,
                 flow: str = "source_to_target", **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        edge_attr = torch.arange(edge_index.size(1))
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                           value=edge_attr, sparse_sizes=(N, N),
                           is_sorted=False)
        adj = adj.t() if flow == 'source_to_target' else adj
        self.adj = adj.to('cpu')

        self.sizes = sizes
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.samplers = []
        for _node_idx in node_idx:
            if not isinstance(_node_idx, torch.Tensor):
                _node_idx = torch.tensor(_node_idx)
            data_loader = torch.utils.data.DataLoader(_node_idx.view(-1).tolist(), 
                                                      collate_fn=self.sample, 
                                                      **kwargs)
            
            self.samplers.append(data_loader)
            
    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs: List[Adj] = []

        n_id = batch
        for size in self.sizes:
            adj, n_id = self.adj.sample_adj(n_id, size, replace=False)
            if self.flow == 'source_to_target':
                adj = adj.t()
            row, col, e_id = adj.coo()
            size = adj.sparse_sizes()
            edge_index = torch.stack([row, col], dim=0)

            adjs.append(Adj(edge_index, e_id, size))

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)