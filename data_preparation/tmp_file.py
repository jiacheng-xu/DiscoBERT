# import os
# files = os.listdir("/datadrive/data/dailymail/chunk")
# os.chdir("/datadrive/data/dailymail/chunk")
# for f in files:
#     f_rest = ".".join(f.split(".")[1:])
#     os.rename(f, "dailymail."+f_rest)

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(g.selfloop_edges())
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask
import time
import numpy as np

g, features, labels, mask = load_cora_data()
print()