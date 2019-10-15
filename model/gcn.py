# Graph Conv and Relational Graph Conv
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import FromParams

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce_sum = fn.sum(msg='m', out='h')
gcn_reduce_max = fn.max(msg='m', out='h')
# gcn_reduce_u_mul_v = fn.u_mul_v('m', 'h')
from depricated.archival_gnns import GraphEncoder


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce_sum)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


from overrides import overrides
import torch
from torch.nn import Dropout

from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.nn.activations import Activation


class GCNNet(nn.Module):
    def __init__(self, hdim: int = 768, nlayers: int = 2, dropout_prob: int = 0.1):
        super(GCNNet, self).__init__()
        # self.gcns = nn.ModuleList([GCN(hdim, hdim, F.relu) for i in range(nlayers)])
        self._gcn_layers = []
        self._feedfoward_layers: List[FeedForward] = []
        self._layer_norm_layers: List[LayerNorm] = []
        self._feed_forward_layer_norm_layers: List[LayerNorm] = []
        feedfoward_input_dim, feedforward_hidden_dim, hidden_dim = hdim, hdim, hdim
        for i in range(nlayers):
            feedfoward = FeedForward(feedfoward_input_dim,
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)

            # Note: Please use `ModuleList` in new code. It provides better
            # support for running on multiple GPUs. We've kept `add_module` here
            # solely for backwards compatibility with existing serialized models.
            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedfoward_layers.append(feedfoward)

            feedforward_layer_norm = LayerNorm(feedfoward.get_output_dim())
            self.add_module(f"feedforward_layer_norm_{i}", feedforward_layer_norm)
            self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

            gcn = GCN(hdim, hdim, F.relu)
            self.add_module(f"gcn_{i}", gcn)
            self._gcn_layers.append(gcn)

            layer_norm = LayerNorm(hdim)
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

            feedfoward_input_dim = hidden_dim

        self.dropout = Dropout(dropout_prob)
        self._input_dim = hdim
        self._output_dim = hdim

    def forward(self, g, features):
        output = features
        # for i, _gcn in enumerate(self.gcns):
        #     x = _gcn(g, x)
        #     feedforward = getattr(self, f"feedforward_{i}")
        #     feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
        #     layer_norm = getattr(self, f"layer_norm_{i}")
        # output = g,x
        for i in range(len(self._gcn_layers)):
            # It's necessary to use `getattr` here because the elements stored
            # in the lists are not replicated by torch.nn.parallel.replicate
            # when running on multiple GPUs. Please use `ModuleList` in new
            # code. It handles this issue transparently. We've kept `add_module`
            # (in conjunction with `getattr`) solely for backwards compatibility
            # with existing serialized models.
            gcn = getattr(self, f"gcn_{i}")
            feedforward = getattr(self, f"feedforward_{i}")
            feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
            layer_norm = getattr(self, f"layer_norm_{i}")
            cached_input = output
            # Project output of attention encoder through a feedforward
            # network and back to the input size for the next layer.
            # shape (batch_size, timesteps, input_size)
            feedforward_output = feedforward(output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                # First layer might have the wrong size for highway
                # layers, so we exclude it here.
                feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output = gcn(g,
                                   feedforward_output)
            output = layer_norm(self.dropout(attention_output) + feedforward_output)

        return output


from typing import List


@GraphEncoder.register("gcn")
class GCN_layers(GraphEncoder, torch.nn.Module, FromParams):

    def __init__(self, hdim: int = 768, nlayers=2
                 ):
        super(GCN_layers, self).__init__()
        self.GCNNet = GCNNet(hdim, nlayers)

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
