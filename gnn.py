"""
This is an OGB's official example code 
https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/products
"""

import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
##############################################
import time
from logger import Logger, Timer
##############################################
from model.gcn import GCN, SAGE
from utils.train_utils import train, test
##############################################

def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products',
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)
    timer = Timer(args.runs, 1, args)
    
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            ############ Train ##########################
            start_t = time.time()
            loss = train(model, data, train_idx, optimizer)
            timer.add_server_time(run, time.time()-start_t)
            ############ Test ###########################
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()

    torch.save(torch.tensor(logger.results), 'results_gnn.pt')
    torch.save(torch.tensor(timer.server_time), 'server_time_gnn.pt')
    
if __name__ == "__main__":
    main()
