from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F

from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.metrics import Accuracy

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv

from pytorch_lightning import LightningModule


class GNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int, heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'gat':
            self.convs.append(
                GATConv(in_channels, hidden_channels // heads, heads))
            self.skips.append(Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels // heads, heads))
                self.skips.append(Linear(hidden_channels, hidden_channels))

        elif self.model == 'graphsage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]
            x = self.convs[i]((x, x_target), adj_t)
            if self.model == 'gat':
                x = x + self.skips[i](x_target)
                x = F.elu(self.norms[i](x))
            elif self.model == 'graphsage':
                x = F.relu(self.norms[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]
