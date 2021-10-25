"""
# Without subgraph approx
Run 0: accuracy 66.25
Run 1: accuracy 66.26
Run 2: accuracy 66.43
66.31±0.08

# With subgraph approx
Run 0: accuracy 66.60
Run 1: accuracy 66.27
Run 2: accuracy 66.20
66.36 ± 0.18
"""

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborSampler

from logger import Logger, Timer
############################################################################################
import os
import copy
import time
import pickle
import numpy as np
import argparse
############################################################################################
from utils.model_avg import model_average, part_model_periodic_avg, model_divergence

from model.gnn_pytorch import GNN
from utils.data_loader_pytorch import MAG240M, sparse_tensor_to_edge_indices

from utils.mini_batch_neighbor_sampling import NeighborSamplerSharedGraphDiffIndex
############################################################################################
ROOT = './dataset'


def get_args():
    ###############################################################
    parser = argparse.ArgumentParser(description='OGBN-MAG240M (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model', type=str,
                        default='graphsage', choices=['gat', 'graphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--runs', type=int, default=3)
    ###############################################################
    parser.add_argument('--avg_per_num_epoch', type=int, default=1)
    parser.add_argument('--num_parts', type=int, default=16)
    parser.add_argument('--local_steps', type=int, default=20)
    parser.add_argument('--use_cut_edges', action='store_true')
    ###############################################################
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
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
    total_acc = 0

    for _ in range(num_iters):
        try:
            batch_size, n_id, adjs = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch_size, n_id, adjs = next(train_iterator)

        adjs = [adj.adj_t.to(device) for adj in adjs]

        optimizer.zero_grad()
        loss, acc = model.training_step(x[n_id].float(),
                                        y[n_id[:batch_size]], adjs)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += float(acc)

    loss = total_loss / num_iters
    acc = total_acc / num_iters
    return loss, acc


@torch.no_grad()
def test(model, x, y, eval_loader, device):
    all_pred = []
    all_targ = []
    
    model.eval()

    # valid
    for batch_size, n_id, adjs in eval_loader:
        adjs = [adj.adj_t.to(device) for adj in adjs]
        
        all_pred.append(model(x[n_id].float(), adjs).softmax(dim=-1).cpu())
        all_targ.append(y[n_id[:batch_size]].cpu())
        
    valid_acc = model.metric(torch.cat(all_pred), torch.cat(all_targ)).item()

    return valid_acc


def main():
    ##############################################
    args, device = get_args()

    logger = Logger(args.runs, args)
    timer = Timer(args.runs, args.num_parts, args)

    data = MAG240M(ROOT, args.num_parts, True)

    ##############################################
    # for local training
    if args.use_cut_edges:
        node_idx = [part[train_part] for part, train_part in zip(data.parts, data.train_parts)]
        local_samplers = NeighborSamplerSharedGraphDiffIndex(sparse_tensor_to_edge_indices(data.adj_t),
                                                             node_idx=node_idx,
                                                             sizes=args.sizes,
                                                             batch_size=1024,
                                                             shuffle=True,
                                                             num_workers=4)
    else:
        local_samplers = []
        for part_id in range(args.num_parts):
            part = data.parts[part_id]
            train_part = data.train_parts[part_id]

            local_sampler = NeighborSampler(data.adj_t[part, :][:, part],
                                            node_idx=train_part,
                                            sizes=args.sizes,
                                            batch_size=1024,
                                            shuffle=True,
                                            num_workers=4)

            local_samplers.append(local_sampler)

    # for evaluation
    eval_loader = NeighborSampler(data.adj_t,
                                  node_idx=data.val_idx,
                                  sizes=args.sizes,
                                  batch_size=1024,
                                  shuffle=False,
                                  num_workers=4)

    ##############################################
    device = get_device(args)

    model = GNN(args.model, data.num_features,
                data.num_classes, args.hidden_channels,
                num_layers=len(args.sizes), dropout=args.dropout)

    model = model.to(device)

    data.x = data.x.to(device)
    data.y = data.y.to(device)

    ##############################################
    ##############################################
    ##############################################
    model_dist = []

    for run in range(args.runs):
        model.reset_parameters()
        best_valid_acc = -1
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
            loss, train_acc = [], []
            epoch_start_time = time.time()
            for part_id in range(args.num_parts):
                start_t = time.time()

                if args.use_cut_edges:
                    loss_, acc_ = train(part_model[part_id], 
                                        data.x, 
                                        data.y, 
                                        part_optimizer[part_id], device,
                                        local_samplers.samplers[part_id], 
                                        num_iters=args.local_steps)
                else:
                    loss_, acc_ = train(part_model[part_id], 
                                        data.x[data.parts[part_id]], 
                                        data.y[data.parts[part_id]], 
                                        part_optimizer[part_id], device,
                                        local_samplers[part_id], 
                                        num_iters=args.local_steps)

                timer.add_local_time(run, part_id, time.time()-start_t)
                loss.append(loss_)
                train_acc.append(acc_)

            loss = sum(loss)/len(loss)
            train_acc  = sum(train_acc)/len(train_acc)
            print('Epoch: ', epoch, ', Time: ', time.time()-epoch_start_time)

            ############ Test ###########################
            if epoch % args.avg_per_num_epoch == 0:    
                avg_model = model_average(part_model)
                part_model = part_model_periodic_avg(part_model, avg_model)

                valid_acc  = test(avg_model, data.x, data.y, eval_loader, device)

                logger.add_result(run, [train_acc, valid_acc, valid_acc])

                print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch + 1:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% ')
                
                if valid_acc > best_valid_acc:
                    torch.save(avg_model.cpu().state_dict(),
                            'best_valid_model_run_%d_subgraph_approx.pt'%run)
                    best_valid_acc = valid_acc

        logger.print_statistics(run)        

    logger.print_statistics()

    res_id = 'mb_ns_w_cut_edge_subgraph_approx' if args.use_cut_edges else 'mb_ns_wo_cut_edge_subgraph_approx'

    log_results = {
        'args': args,
        'results': logger.results,
        'model_dist': model_dist,
        'time': timer.local_time
    }

    with open('log_results_%s.pkl' % res_id, 'wb') as f:
        pickle.dump(log_results, f)

    ######### Final Testing ################################
    print('Evaluation ....')
    eval_loader = NeighborSampler(data.adj_t,
                                  node_idx=data.val_idx,
                                  sizes=[160] * len(args.sizes),
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4)
    acc_results = []
    for run in range(args.runs):
        model.load_state_dict(torch.load('best_valid_model_run_%d_subgraph_approx.pt'%run))
        acc_result  = test(model, data.x, data.y, eval_loader, device)
        acc_results.append(acc_result*100)
        print('Run %d: accuracy %.2f'%(run, acc_result*100))
    print('%.2f ± %.2f'%(np.mean(acc_results), np.std(acc_results)))

if __name__ == "__main__":
    main()
