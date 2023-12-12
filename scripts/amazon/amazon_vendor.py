import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

import time

import dgl
from dgl.data import DGLDataset
from dgl.nn.pytorch import GraphConv
import networkx as nx
import os

class AmazonDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="amazon")
    def process(self):
        with open("/local/jungi/com-amazon.ungraph.txt") as f:
            lines = f.read().splitlines()
            edges = []
            node_id = 0
            for line in tqdm(lines[4:]):
                src, dst = line.split()
                edges.append([int(src), int(dst)])

            edges = torch.tensor(edges)

        srcs = edges[:,0].squeeze() - 1
        dsts = edges[:,1].squeeze() - 1
        max_ = max(srcs.max().item(), dsts.max().item())

        num_nodes = max_ + 1
        self.num_classes = 58
        identity = torch.arange(num_nodes)

#        idx = torch.full((max_+1,), -1)
#        cur_node =-1
#        for i in tqdm(range(srcs.shape[0])):
#            id = srcs[i]
#            if(idx[id].item()==-1):
#                cur_node += 1
#                idx[id] = cur_node
#            srcs[i] = cur_node

#        dsts = idx[dsts]
#        print(srcs.max())
#        print(dsts.max())

        edges_src = torch.cat((srcs,dsts,identity),0)
        edges_dst = torch.cat((dsts,srcs,identity),0)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
        self.graph.ndata["feat"] = torch.randn((num_nodes, 128))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


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
from dgl import save_graphs, load_graphs

#dataset = AmazonDataset()
#
#g = dataset[0]
#save_graphs("/local/jungi/dgl/amazon", g)
g = load_graphs("/local/jungi/dgl/amazon")[0][0]


# referenced from Cluter-GCN (KDD'19)
num_layers = 1      #2 layer
num_hidden = 128    #128 hidden units
nfeat = g.ndata["feat"]
infeat_dim = nfeat.shape[1]
num_classes = 58
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
torch_model.eval()

# Vanilla version: DGL backend
os.system("rm out.txt")
os.system("touch out.txt")
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
#with torch.no_grad():
#    model = torch.jit.trace(model, (nfeat,), check_trace = False, strict = False)
#    model = torch.jit.freeze(model)
#    model = torch.jit.optimize_for_inference(model)
#    
#    cost=[]
#    for i in range(3):
#       # Warmup
#        for _ in range(5):
#            model(nfeat)
#        start = time.time()
#        model(nfeat)
#        end = time.time()
#        cost.append((end-start)*1000)
#    
#        print('Latency of iter %d is %f ms'% (i, cost[-1]))
#
#print('[Pytorch mode]')
#print("Latency is %f ms" %(sum(cost)/len(cost)))

# Profiling ##################################
#import cProfile 
#import pstats
#from pstats import SortKey
#
#vendor_stat_file = "/dot/scratch/share/mlcomp/tvm/gnn/ogbn-products/dgl_vendor.stats"
#with torch.no_grad():
#    cProfile.run("model(nfeat)", vendor_stat_file)
#
#p = pstats.Stats(vendor_stat_file)
#p.sort_stats(SortKey.CUMULATIVE).print_stats()
