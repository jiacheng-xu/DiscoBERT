import torch
import torch.nn as nn
import torch.nn.functional as F

from model.archival_gnns import GraphEncoder


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


import dgl.function as fn
import torch
import logging, itertools, dgl, random, torch, tempfile
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import FromParams
from dgl import DGLGraph
from overrides import overrides
from allennlp.modules.masked_layer_norm import MaskedLayerNorm

from allennlp.modules.feedforward import FeedForward

from allennlp.modules.layer_norm import LayerNorm


@GraphEncoder.register("gat")
class GCN_layers(GraphEncoder, torch.nn.Module, FromParams):

    def __init__(self, hdim: int = 768
                 ):
        super(GCN_layers, self).__init__()
        self.gat = GAT

    def transform_sent_rep(self, sent_rep, sent_mask, meta_field, key):
        init_graphs = self.convert_sent_tensors_to_graphs(sent_rep, sent_mask, meta_field, key)
        unpadated_graphs = []
        for g in init_graphs:
            updated_graph = self.forward(g)
            unpadated_graphs.append(updated_graph)
        recovered_sent = torch.stack(unpadated_graphs, dim=0)
        assert recovered_sent.shape == sent_rep.shape
        return recovered_sent

    @overrides
    def forward(self, g):
        h = g.ndata['h']

        out = self.GCNNet.forward(g, features=h)
        # return g, g.ndata['h'], hg  # g is the raw graph, h is the node rep, and hg is the mean of all h
        return out

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False
