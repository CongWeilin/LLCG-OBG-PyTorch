import torch

def prop_row_norm(data):
    adj_t = data.adj_t
    
    # global norm
    adj_t_parts = []
    for part in data.parts:
        adj_t_part = adj_t[part, :][:, part]
        adj_t_parts.append(adj_t_part)
    data.adj_t_parts = adj_t_parts
    
    return data


def prop_sym_norm(data):
    # add identity
    adj_t = data.adj_t.set_diag()

    # create local
    adj_t_parts = []
    for part in data.parts:
        adj_t_part = adj_t[part, :][:, part]
        deg_part = adj_t_part.sum(dim=1).to(torch.float)
        deg_inv_sqrt_part = deg_part.pow(-0.5)
        deg_inv_sqrt_part[deg_inv_sqrt_part == float('inf')] = 0
        adj_t_part = deg_inv_sqrt_part.view(-1, 1) * adj_t_part * deg_inv_sqrt_part.view(1, -1)
        adj_t_parts.append(adj_t_part)
    data.adj_t_parts = adj_t_parts

    # global norm
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t
    
    return data