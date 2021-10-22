import copy
import torch

def model_average(models):
    state_dicts = [model.state_dict() for model in models]
    averaged_state_dict = average_weights(state_dicts)
    
    model = copy.deepcopy(models[0])
    model.load_state_dict(averaged_state_dict)
    return model

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def part_model_periodic_avg(part_model, avg_model):
    num_part_model = len(part_model)
    avg_model_state_dict = avg_model.state_dict()
    
    for part_id in range(num_part_model):
        for n, p in part_model[part_id].named_parameters():
            p.data = p.data - (p.data - avg_model_state_dict[n].data)

    return part_model

def model_divergence(models):
    state_dicts = [model.state_dict() for model in models]
    num_models = len(state_dicts)
    
    model_divergence_ = []
    for i in range(num_models-1):
        for j in range(i+1, num_models):
            model_dist_ = 0
            for n, p in models[0].named_parameters():
                model_dist_ += (state_dicts[i][n].data - state_dicts[j][n].data).norm(2).item()
            model_divergence_ += [model_dist_]
            
    return model_divergence_

def assign_model_weight(from_model, to_model):
    from_model_state_dict = from_model.state_dict()
    for n, p in to_model.named_parameters():
        p.data = p.data - (p.data - from_model_state_dict[n].data)
    return to_model