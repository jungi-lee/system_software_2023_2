import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import time

import dgl
from dgl.nn.pytorch import GraphConv
import networkx as nx
import os

import warnings
warnings.filterwarnings('ignore')

#from memory_profiler import profile

# Jungi: Using dgl version of 0.8.0
class GCN(nn.Module):
    def __init__(self, g, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(GCN, self).__init__()
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(n_infeat, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_classes))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        return h

# Load dataset
from dgl import AddSelfLoop
from dgl.data import RedditDataset

transform = (
    AddSelfLoop()
)  # by default, it will first remove self-loops to prevent duplication

dataset = RedditDataset(raw_dir="/local/jungi/dgl", transform=transform)

# there is only one graph in Node Property Prediction datasets
g = dataset[0]

# Checkout graph structure: Although g is typed as heterogeneous, it is actually homogeneous graph
#print("Type of the graph imported from ogb: " + str(type(g)))
#print("All node type names: "+ str(g.ntypes))
#print("All edge type names: "+ str(g.etypes))
#print("All source node type names: "+ str(g.srctypes))
#print("All dest node type names: "+ str(g.dsttypes))
#print("All canonical edge types: "+ str(g.canonical_etypes))
#print("<Nodes>")
#print("Number of nodes: " + str(g.num_nodes()))
#print(g.nodes())
#print("<Edges>")
#print("Number of edges: " + str(g.num_edges()))
#print(g.edges())

# Organize parameters
# Configuration of original GCN paper, Cluster-GCN
num_layers = 1      #2 layer
num_hidden = 128    #128 hidden units
nfeat = g.ndata["feat"]
#print("nfeat\n" + str(nfeat))
infeat_dim = nfeat.shape[1]
num_classes = dataset.num_classes
#print("g.ndata[\"feat\"].shape: " + str(nfeat.shape)+str(type(infeat_dim)))
#print("num_classes: " + str(num_classes) + str(type(num_classes)))

torch_model = GCN(g, infeat_dim, num_hidden, num_classes, num_layers, F.relu)

# Initialize weight with some dummy data
from collections import OrderedDict

rand_params = OrderedDict()
for i in range(len(torch_model.layers)):
    w = torch_model.layers[i].weight
    rand_params["layers.%d.weight" % (i)] = torch.rand(tuple(w.shape), dtype = w.dtype)
    b = torch_model.layers[i].bias
    rand_params["layers.%d.bias" %(i)] = torch.rand(tuple(b.shape), dtype = b.dtype)

torch_model.load_state_dict(rand_params)

# Evaluation ##################################
import sys

torch_model.eval()
#torch_model(nfeat)
# Vanilla version: DGL backend

os.system("rm out.txt")
os.system("touch out.txt")

#with torch.no_grad():
#    with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
#        cost=[]
#        for i in range(1):
#           # Warmup
#            for _ in range(3):
#                torch_model(nfeat)
#            start = time.time()
#            torch_model(nfeat)
#            end = time.time()
#            cost.append((end-start)*1000)

num_iters = 3
with torch.no_grad():
    for i in range(num_iters):
        torch_model(nfeat)

k_blocking_param = int(os.environ.get("K_BLOCKING_PARAM"))
m_blocking_param = int(os.environ.get("M_BLOCKING_PARAM"))

with open("out.txt") as f:
    lines = f.readlines()

spmm_all=[]
blocking_all=[]
for line in lines:
    l = line.strip().split(' ')
    blocking_all.append(float(l[0]))
    spmm_all.append(float(l[1]))

spmm_iter=[]
blocking_iter=[]
for i in range(num_iters):
    spmm = 0.0
    blocking = 0.0
    for j in range((num_layers+1)*i, (num_layers+1)*(i+1)):
        spmm += spmm_all[j]
        blocking += blocking_all[j]

    spmm_iter.append(spmm)
    blocking_iter.append(blocking)

#print("m_block_factor %d k_block_factor %d spmm_avg_latency %f ms blocking_avg_latency %f ms" %(m_blocking_param, k_blocking_param, sum(spmm_iter)/len(spmm_iter), sum(blocking_iter)/len(blocking_iter)))
print("%f %f " %(sum(spmm_iter)/len(spmm_iter), sum(blocking_iter)/len(blocking_iter)))


# Vendor library: DGL + OneDNN
#import intel_extension_for_pytorch as ipex
#torch.jit.enable_onednn_fusion(True)
#model = ipex.optimize(torch_model)
#
#sys.stdout.flush()
#print('[Vendor Library]')
#sys.stdout.flush()
#with torch.no_grad():
#    model = torch.jit.trace(model, (nfeat,), check_trace = False, strict = False)
#    model = torch.jit.freeze(model)
#    model = torch.jit.optimize_for_inference(model)

#with torch.no_grad():
#    with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
#        cost=[]
#        for i in range(3):
           # Warmup
#            for _ in range(5):
#                model(nfeat)
#            start = time.time()
#            model(nfeat)
#            end = time.time()
#            cost.append((end-start)*1000)
#
#print("Average latency is %f ms" %(sum(cost)/len(cost)))
#sys.stdout.flush()
    
## Profiling ##################################
#import cProfile 
#import pstats
#from pstats import SortKey
#
#vanilla_stat_file = "/dot/scratch/share/mlcomp/tvm/gnn/pubmed/pubmed_vanilla.stats"
#with torch.no_grad():
#    cProfile.run("torch_model(nfeat)", vanilla_stat_file)
#
#p = pstats.Stats(vanilla_stat_file)
#p.sort_stats(SortKey.CUMULATIVE).print_stats()
#
#vendor_stat_file = "/dot/scratch/share/mlcomp/tvm/gnn/pubmed/pubmed_vendor.stats"
#with torch.no_grad():
#    cProfile.run("model(nfeat)", vendor_stat_file)
#
#p = pstats.Stats(vendor_stat_file)
#p.sort_stats(SortKey.CUMULATIVE).print_stats()
