"""
Code is modified from OGB's official example code 
https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/products
"""

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from logger import Logger, Timer
############################################################################################
import os
import copy
import time
import pickle

import argparse
############################################################################################
from utils.model_avg import model_average, part_model_periodic_avg, model_divergence, assign_model_weight
from utils.args import get_args

from model.mb_ns_graphsage import SAGE
from utils.data_loader import load_dataset, sparse_tensor_to_edge_indices
from utils.graph_prop_preprocess import prop_row_norm, prop_sym_norm

from torch_geometric.loader import NeighborSampler
from utils.mini_batch_neighbor_sampling import NeighborSamplerSharedGraphDiffIndex
############################################################################################

def get_args():
    ###############################################################
    parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=3)
    ###############################################################
    parser.add_argument('--avg_per_num_epoch', type=int, default=1)
    parser.add_argument('--num_parts', type=int, default=16)
    parser.add_argument('--local_steps', type=int, default=20)
    parser.add_argument('--use_cut_edges', action='store_true')
    parser.add_argument('--partition_method', type=int, default=2)
    ###############################################################
    args = parser.parse_args()
    print(args)
    return args, get_device(args)

def get_device(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    return device

def train(model, x, y, optimizer, device, train_loader, num_iters):
    model.train()
    train_iterator = iter(train_loader)
    
    total_loss = 0
    
    for _ in range(num_iters):
        try:
            batch_size, n_id, adjs = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch_size, n_id, adjs = next(train_iterator)
            
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)

    loss = total_loss / num_iters
    
    return loss


@torch.no_grad()
def test(model, x, y, split_idx, evaluator, device, subgraph_loader):
    model.eval()

    out = model.inference(x, subgraph_loader, device)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

def main():
    ##############################################
    args, device = get_args()
    
    logger = Logger(args.runs, args)
    timer = Timer(args.runs, args.num_parts, args)

    dataset, data, split_idx, evaluator = load_dataset(dataset_name     = 'ogbn-products',
                                                       num_parts        = args.num_parts,
                                                       partition_method = args.partition_method)
    ##############################################
    # for local training
    if args.use_cut_edges:
        node_idx = [part[train_part] for part, train_part in zip(data.parts, data.train_parts)]
        local_samplers = NeighborSamplerSharedGraphDiffIndex(sparse_tensor_to_edge_indices(data.adj_t), 
                                                             node_idx   = node_idx,
                                                             sizes      = [15, 10, 5], 
                                                             batch_size = 1024,
                                                             shuffle    = True, 
                                                             num_workers=4)
    else:
        local_samplers = []
        for part_id in range(args.num_parts):
            part       = data.parts[part_id]
            train_part = data.train_parts[part_id]

            local_sampler = NeighborSampler(sparse_tensor_to_edge_indices(data.adj_t[part, :][:, part]), 
                                            node_idx   = train_part,
                                            sizes      = [15, 10, 5], 
                                            batch_size = 1024,
                                            shuffle    = True, 
                                            num_workers=4)

            local_samplers.append(local_sampler)
            
    # for evaluation
    subgraph_loader = NeighborSampler(sparse_tensor_to_edge_indices(data.adj_t), 
                                      node_idx=None, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=4)
    
    ##############################################
    device = get_device(args)

    model = SAGE(dataset.num_features, args.hidden_channels, dataset.num_classes, args.num_layers)
    model = model.to(device)

    data = data.to(device)
    ##############################################
    
    model_dist = []

    for run in range(args.runs):
        model.reset_parameters()

        ##############################################
        part_model = []
        part_optimizer = []
        for part_id in range(args.num_parts):
            part_model.append(copy.deepcopy(model))
            part_optimizer.append(torch.optim.Adam(part_model[-1].parameters(), lr=args.lr))
        ##############################################

        for epoch in range(args.epochs):
            # check model divergence to see if method help
            model_dist += [model_divergence(part_model)]

            ############ Train ##########################
            loss = []
            epoch_start_time = time.time()
            for part_id in range(args.num_parts):
                start_t = time.time()

                if args.use_cut_edges:
                    loss_ = train(part_model[part_id], 
                                  data.x, 
                                  data.y.squeeze(), 
                                  part_optimizer[part_id], device,
                                  local_samplers.samplers[part_id], 
                                  num_iters=args.local_steps)
                else:
                    loss_ = train(part_model[part_id], 
                                  data.x[data.parts[part_id]], 
                                  data.y[data.parts[part_id]].squeeze(), 
                                  part_optimizer[part_id], device,
                                  local_samplers[part_id], 
                                  num_iters=args.local_steps)

                timer.add_local_time(run, part_id, time.time()-start_t)
                loss.append(loss_)
            loss = sum(loss)/len(loss)
            print('Epoch: ', epoch, ', Time: ', time.time()-epoch_start_time)
            
            ############ Test ###########################
            if epoch % args.avg_per_num_epoch == 0:    
                avg_model = model_average(part_model)
                part_model = part_model_periodic_avg(part_model, avg_model)
                result  = test(avg_model, 
                               data.x, 
                               data.y.squeeze(), 
                               split_idx, 
                               evaluator, 
                               device, subgraph_loader)
                logger.add_result(run, result)

                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch + 1:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()
    
    res_id = 'mb_ns_w_cut_edge_subgraph_approx' if args.use_cut_edges else 'mb_ns_wo_cut_edge_subgraph_approx'
    
    log_results = {
        'args': args,
        'results': logger.results,
        'model_dist': model_dist, 
        'time': timer.local_time
    }
    
    with open('log_results_%s.pkl'%res_id, 'wb') as f:
        pickle.dump(log_results, f)
    
if __name__ == "__main__":
    main()
