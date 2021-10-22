import torch
import torch.nn.functional as F

from torch_sparse import matmul
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

##############################################
##############################################
##############################################
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)
        
    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        x = self.lin(x)
        return x
    
    def forward_without_linear(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        return x

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
    
##############################################
##############################################
##############################################
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
    
    def forward_with_last_layer_feats(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1].forward_without_linear(x, adj_t)
        x = self.convs[-1].lin(h)
        return torch.log_softmax(x, dim=-1), h.detach()
    
    def forward_with_last_lin(self, h):
        x = self.convs[-1].lin(h)
        return torch.log_softmax(x, dim=-1)
        
##############################################
##############################################
##############################################
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_l = torch.nn.Linear(in_channels, out_channels)
        self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        
    def forward(self, x, edge_index):
        x_prop = self.propagate(edge_index, x=x)
        x = self.lin_l(x_prop) + self.lin_r(x)
        return x
    
    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
##############################################
##############################################
##############################################
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)