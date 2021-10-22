
import pickle
import os
import torch

##############################################
##############################################
##############################################
def get_label_partitions(dataset, num_clusters, data, root):

    save_parts_path = os.path.join(root, 'dataset/%s_%d_label_partition.pkl'%(dataset, num_clusters))

    if os.path.exists(save_parts_path):
        all_part_labels = pickle.load(open(save_parts_path, 'rb'))
    else:
        
        y = data.y.squeeze()
        num_labels = len(torch.unique(y))
        all_labels = torch.randperm(num_labels)
        all_label_chunks = torch.chunk(all_labels, num_clusters)
            
        all_part_labels = []

        for part_id in range(num_clusters):
            print('part: ', part_id, ', labels: ', all_label_chunks[part_id])
            
            part_labels = []
            for _label in all_label_chunks[part_id]:
                part_labels.append(torch.where(y==_label)[0])
            part_labels = torch.cat(part_labels).numpy()
            all_part_labels.append(part_labels)

        with open(save_parts_path, 'wb') as f:
            pickle.dump(all_part_labels, f)
            
    return all_part_labels
