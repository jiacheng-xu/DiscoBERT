# Graph Conv and Relational Graph Conv
import itertools
import torch
from typing import List, Union

import dgl
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import FromParams
from allennlp.common import Registrable
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from overrides import overrides


class GraphEncoder(_EncoderBase, Registrable):
    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def is_bidirectional(self):
        raise NotImplementedError
    #
    def convert_sent_tensors_to_graphs(self, sent, sent_mask, meta_field, key):
        batch_size, max_sent_num, hdim = sent.shape
        effective_length = torch.sum(sent_mask, dim=1).long().tolist()
        graph_bag = []
        for b in range(batch_size):
            this_sent = sent[b]  # max_sent, hdim
            this_len = effective_length[b]
            graph_seed = meta_field[b][key]  # List of tuples
            G = dgl.DGLGraph()
            G.add_nodes(max_sent_num)
            # fc_src = [i for i in range(this_len)] * this_len
            # fc_tgt = [[i] * this_len for i in range(this_len)]
            # fc_tgt = list(itertools.chain.from_iterable(fc_tgt))
            fc_src = [x[0] for x in graph_seed]
            fc_tgt = [x[1] for x in graph_seed]
            G.add_edges(fc_src, fc_tgt)
            G.ndata['h'] = this_sent  # every node has the parameter
            graph_bag.append(G)
        return graph_bag


@GraphEncoder.register("easy_graph_encoder")
class EasyGraph(GraphEncoder, torch.nn.Module, FromParams):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, List[int]],
                 dropout=0.1):
        super().__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore

        self._activations = [torch.nn.functional.relu] * num_layers
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]

        self.lin = torch.nn.Linear(self._output_dim, self._output_dim)
        self.ln = MaskedLayerNorm(size=hidden_dims[0])

    def transform_sent_rep(self, sent_rep, sent_mask, graphs):
        # LayerNorm(x + Sublayer(x))
        output = sent_rep

        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            mid = layer(output)  # output: batch, seq, feat
            mid = mid.permute(0, 2, 1)  # mid: batch, feat, seq

            nex = torch.bmm(mid, graphs)
            output = dropout(activation(nex))
            output = output.permute(0, 2, 1)  # mid: batch, seq, feat
        middle = sent_rep + self.lin(output)
        output = self.ln.forward(middle, sent_mask)
        return output


@GraphEncoder.register("old_gcn")
class GCN_layers(GraphEncoder, torch.nn.Module, FromParams):

    def __init__(self, input_dims: List[int],
                 num_layers: int,
                 hidden_dims: Union[int, List[int]],
                 activations='relu'):
        super(GCN_layers, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        # TODO remove hard code relu
        activations = [torch.nn.functional.tanh] * num_layers
        assert len(input_dims) == len(hidden_dims) == len(activations) == num_layers
        gcn_layers = []
        for layer_input_dim, layer_output_dim, activate in zip(input_dims, hidden_dims, activations):
            gcn_layers.append(GCN(layer_input_dim, layer_output_dim, activate))
        self.layers = nn.ModuleList(gcn_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dims[0]
        self.ln = LayerNorm(hidden_dims[0])
        self._mlp = FeedForward(hidden_dims[0], 1, hidden_dims[0], torch.nn.functional.sigmoid)

    def transform_sent_rep(self, sent_rep, sent_mask, sent_graph):
        init_graphs = self.convert_sent_tensors_to_graphs(sent_rep, sent_mask)
        unpadated_graphs = []
        for g in init_graphs:
            updated_graph = self.forward(g)
            unpadated_graphs.append(updated_graph)
        recovered_sent = torch.stack(unpadated_graphs, dim=0)
        assert recovered_sent.shape == sent_rep.shape
        return recovered_sent

    def convert_sent_tensors_to_graphs(self, sent, sent_mask):
        batch_size, max_sent_num, hdim = sent.shape
        effective_length = torch.sum(sent_mask, dim=1).long().tolist()
        graph_bag = []
        for b in range(batch_size):
            this_sent = sent[b]  # max_sent, hdim
            # this_mask = sent_mask[b]
            this_len = effective_length[b]

            G = dgl.DGLGraph()
            G.add_nodes(max_sent_num)
            fc_src = [i for i in range(this_len)] * this_len
            fc_tgt = [[i] * this_len for i in range(this_len)]
            fc_tgt = list(itertools.chain.from_iterable(fc_tgt))

            G.add_edges(fc_src, fc_tgt)
            G.ndata['h'] = this_sent  # every node has the parameter
            graph_bag.append(G)
        return graph_bag

    @overrides
    def forward(self, g):
        # h = g.in_degrees().view(-1, 1).float()
        h = g.ndata['h']
        output = h
        for conv in self.layers:
            output = conv(g, output)
            print(output)
        norm_output = self.ln(h + output)
        # print(norm_output)
        # m = self._mlp(norm_output)
        # h = self.ln(norm_output + m)
        h = norm_output
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        # return g, g.ndata['h'], hg  # g is the raw graph, h is the node rep, and hg is the mean of all h
        return g.ndata['h']

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False


def discourse_oracle(disco_txt, ):
    # oracle labels
    docs = [disc.get_readable_words_as_list() for disc in disco_bag]

    # rewrite the docs to accomodate the dependency
    modified_docs_w_deps = []
    oracle_inclusion = []
    for idx, disco in enumerate(disco_bag):
        # tmp_txt, tmp_oracle_inclusion = copy.deepcopy(docs[idx]),[idx]
        tmp_txt, tmp_oracle_inclusion = [], []
        if disco.dep != []:
            for _d in disco.dep:
                if _d < len(docs):
                    tmp_txt += docs[_d]
                    tmp_oracle_inclusion.append(_d)
            tmp_txt += copy.deepcopy(docs[idx])
            tmp_oracle_inclusion.append(idx)
            modified_docs_w_deps.append(" ".join(tmp_txt))
            oracle_inclusion.append(tmp_oracle_inclusion)
        else:
            modified_docs_w_deps.append(
                " ".join(docs[idx])
            )
            oracle_inclusion.append([idx])

    yangliu_label = original_greedy_selection([x.split(" ") for x in modified_docs_w_deps], summary, 5)
    # oracle_ids = greedy_selection(modified_docs_w_deps, summary, oracle_size)
    return yangliu_labelf
