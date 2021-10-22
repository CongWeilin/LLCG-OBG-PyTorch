"""
Code is modified from OGB's official example code 
https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/products
"""

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from logger import Logger, Timer

import os
import copy
import time

############################################################################################
from utils.model_avg import model_average, part_model_periodic_avg, model_divergence, assign_model_weight
from utils.args import get_args

from model.full_batch_gcn import GCN, SAGE
from utils.train_utils import train, test, train_partition
from utils.data_loader import load_dataset
from utils.graph_prop_preprocess import prop_row_norm, prop_sym_norm

############################################################################################

def main():
    args, device = get_args()

    logger = Logger(args.runs, args)
    timer = Timer(args.runs, args.num_parts, args)

    dataset, data, split_idx, evaluator = load_dataset(dataset_name = 'ogbn-products',
                                                       num_parts    = args.num_parts)

    if args.use_sage:
        model = SAGE(data.num_features, 
                     args.hidden_channels,
                     dataset.num_classes, 
                     args.num_layers,
                     args.dropout)
        data = prop_row_norm(data)
    else:
        model = GCN(data.num_features, 
                    args.hidden_channels,
                    dataset.num_classes, 
                    args.num_layers,
                    args.dropout)
        data = prop_sym_norm(data)
    
    model = model.to(device)
    data  = data.to(device)
    
    train_idx = split_idx['train'].to(device)
    
    model_dist = []
    
    for run in range(args.runs):
        model.reset_parameters()
        
        ##############################################
        part_model = []
        part_optimizer = []
        for part_id in range(args.num_parts):
            part_model.append(copy.deepcopy(model))
            part_optimizer.append(torch.optim.Adam(part_model[-1].parameters(), lr=args.lr))
        server_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        ##############################################

        for epoch in range(1, 1 + args.epochs):
            # check model divergence to see if method help
            model_dist += [model_divergence(part_model)]
            
            ###################################################
            ############ Local Train ##########################
            loss = []
            for part_id in range(args.num_parts):
                start_t = time.time() # clock start
                loss_ = train_partition(part_model, data, part_optimizer, part_id, args.use_cut_edges)
                timer.add_local_time(run, part_id, time.time()-start_t) # clock end
                loss.append(loss_)
            loss = sum(loss)/len(loss)
            
            ###################################################
            ############ Local Test ###########################
            avg_model = model_average(part_model)
            if epoch % args.avg_per_num_epoch == 0:
                part_model = part_model_periodic_avg(part_model, avg_model)
            result  = test(avg_model, data, split_idx, evaluator)
            logger.add_result(run, result)
            
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Local Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                
            ###################################################
            ############ Global Train #########################
            if epoch % args.avg_per_num_epoch == 0:
                model = assign_model_weight(avg_model, model)
                start_t = time.time() # clock start
                train(model, data, train_idx, server_optimizer)
                timer.add_server_time(run, time.time()-start_t) # clock end
                part_model = part_model_periodic_avg(part_model, model)
                result  = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)
            
                train_acc, valid_acc, test_acc = result
                print(f'Server Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()

    torch.save(torch.tensor(logger.results), 'results_period_avg_corr.pt')
    torch.save(torch.tensor(model_dist), 'model_dist_period_avg_corr.pt')

    torch.save(torch.tensor(timer.local_time),  'local_time_period_avg_corr.pt')
    torch.save(torch.tensor(timer.server_time), 'server_time_period_avg_corr.pt')
    
if __name__ == "__main__":
    main()
