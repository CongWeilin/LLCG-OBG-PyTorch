# LLCG-OBG-PyTorch
 
This code is modified from [OGB's example code](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py). The code can simulate the environment with any local machines. Code has been tested on a 32GB RAM + RTX 3090 (24GB GPU memory) machine. This includes a naive periodic parameter averaging (PSGD-PA), [periodic parameter averaging with subgraph approximation](https://arxiv.org/abs/2012.04930), and our proposal Learn Locally Correct Globally (LLCG).

## Run code
```
# Run naive periodic parameter averaging (PSGD-PA)
python gnn_with_periodic_avg_mb_ns.py

# Run periodic parameter averaging with subgraph approximation
python gnn_with_periodic_avg_mb_ns_subgraph_approx.py

# Run Learn Locally Correct Globally (LLCG)
python gnn_with_periodic_avg_mb_ns_corr.py
```