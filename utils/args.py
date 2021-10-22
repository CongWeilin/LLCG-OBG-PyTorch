import torch
import argparse

def get_args():
    ###############################################################
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
    ###############################################################
    parser.add_argument('--avg_per_num_epoch', type=int, default=15)
    parser.add_argument('--num_parts', type=int, default=4)
    parser.add_argument('--use_cut_edges', action='store_true')
    ###############################################################
    args = parser.parse_args()
    print(args)
    return args, get_device(args)

def get_device(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    return device