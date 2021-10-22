import torch
import torch.nn.functional as F

##############################################
##############################################
##############################################
def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

##############################################
##############################################
##############################################
def train_partition(model, data, optimizer, part_id, use_cut_edges=False):
    partition = data.parts[part_id]
    train_part = data.train_parts[part_id]
    
    model[part_id].train()

    optimizer[part_id].zero_grad()
    if use_cut_edges:
        out = model[part_id](data.x[partition, :], 
                             data.adj_t_parts[part_id])[train_part]
        loss = F.nll_loss(out, data.y.squeeze(1)[partition][train_part])
    else:
        out = model[part_id](data.x, data.adj_t)[partition][train_part]
        loss = F.nll_loss(out, data.y.squeeze(1)[partition][train_part])
    loss.backward()
    optimizer[part_id].step()
    return loss.item()