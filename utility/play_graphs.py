import torch.nn.functional as F
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
import dgl

# construct graph
num_sent = 9
embedding_dim = 13
hid_dim = 17
sent_rep = torch.FloatTensor(num_sent, embedding_dim).uniform_(-1, 1)

# full connect
G = dgl.DGLGraph()
G.add_nodes(num_sent)
import itertools

fc_src = [i for i in range(num_sent)] * num_sent
fc_tgt = [[i] * num_sent for i in range(num_sent)]
merged = list(itertools.chain.from_iterable(fc_tgt))

G.add_edges(fc_src, merged)
G.ndata['h'] = sent_rep  # every node has the parameter

NUM = num_sent-4
SSSS = torch.FloatTensor(NUM, embedding_dim).uniform_(-1, 1)
g = dgl.DGLGraph()
g.add_nodes(NUM)

fc_src = [i for i in range(NUM)] * NUM
fc_tgt = [[i] * NUM for i in range(NUM)]
merged = list(itertools.chain.from_iterable(fc_tgt))

g.add_edges(fc_src, merged)
g.ndata['h'] = SSSS  # every node has the parameter

print(G)
print(g)
from model.tensor_bert import GCN_layers

net = GCN_layers([embedding_dim, hid_dim], 2,
                 hidden_dims=[hid_dim, hid_dim],
                 activations=[F.relu, F.relu])
print(net)
batch_input = dgl.batch([g,G])
newg, rep, output = net(batch_input)
# print(output.shape)
# print(rep)
# print(rep.shape)
print(rep.shape)
unbatched_rep = dgl.unbatch(rep)
print(unbatched_rep)