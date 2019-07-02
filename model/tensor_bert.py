import logging, itertools, dgl, random, torch, tempfile
from typing import Dict, List, Optional, Union
import numpy as np
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel
from torch.nn.functional import nll_loss
import torch.nn as nn
from torch import autograd
from allennlp.common import FromParams
from utility.learn_dgl import GCN
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common import Registrable
from model.data_reader import CNNDMDatasetReader
# from model.pythonrouge_metrics import RougeStrEvaluation
from model.pyrouge_metrics import PyrougeEvaluation

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from pytorch_pretrained_bert import BertConfig

from model.model_util import *


class GraphEncoder(_EncoderBase, Registrable):
    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def is_bidirectional(self):
        raise NotImplementedError


from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention


@GraphEncoder.register("seq2seq")
class S2S(GraphEncoder, torch.nn.Module, FromParams):
    def __init__(self, seq2seq_encoder: Seq2SeqEncoder):
        super().__init__()
        self.seq2seq = seq2seq_encoder

    def transform_sent_rep(self, sent_rep, sent_mask):
        ouputs = self.seq2seq.forward(sent_rep, sent_mask)
        # print(ouputs.shape)
        return ouputs


@GraphEncoder.register("identity")
class Identity(GraphEncoder, torch.nn.Module, FromParams):
    def __init__(self):
        super(Identity, self).__init__()

    def transform_sent_rep(self, sent_rep, sent_mask):
        return sent_rep


from allennlp.modules.feedforward import FeedForward


@GraphEncoder.register("gcn")
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

    def transform_sent_rep(self, sent_rep, sent_mask):
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
        print(norm_output)
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


from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.span_extractors.span_extractor import SpanExtractor


@Model.register("tensor_bert")
class TensorBertSum(Model):
    def __init__(self, vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 bert_config_file: str,
                 graph_encoder: GraphEncoder,
                 span_extractor: SpanExtractor,
                 trainable: bool = True,
                 use_disco: bool = True,
                 use_coref: bool = False,
                 index: str = "bert",
                 dropout: float = 0.2,
                 tmp_dir: str = '/datadrive/tmp/',
                 pred_length: int = 5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        # super(TensorBertSum, self).__init__(vocab, regularizer)
        super(TensorBertSum, self).__init__(vocab)
        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        self.bert_model.config = BertConfig.from_json_file(bert_config_file)
        self._graph_encoder = graph_encoder

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        in_features = self.bert_model.config.hidden_size
        self._pred_length = pred_length
        self._index = index
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(in_features, 1)
        # self._transfrom_layer = torch.nn.Linear(100, 2)
        # self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.BCELoss(reduction='none')
        self._layer_norm = MaskedLayerNorm(768)
        self._rouge = PyrougeEvaluation(name='rouge', cand_path=tmp_dir, ref_path=tmp_dir, path_to_valid=tmp_dir)
        self._sigmoid = nn.Sigmoid()
        # self.bert_model.requires_grad = False
        initializer(self._classification_layer)
        # xavier_uniform_(self._classification_layer._parameters())
        self._use_disco = use_disco
        if self._use_disco:
            self._span_extractor = span_extractor
        self._use_coref = use_coref

    def transform_sent_rep(self, sent_rep, sent_mask):
        init_graphs = self._graph_encoder.convert_sent_tensors_to_graphs(sent_rep, sent_mask)
        unpadated_graphs = []
        for g in init_graphs:
            updated_graph = self._graph_encoder(g)
            unpadated_graphs.append(updated_graph)
        recovered_sent = torch.stack(unpadated_graphs, dim=0)
        assert recovered_sent.shape == sent_rep.shape
        return recovered_sent

    @overrides
    def forward(self, tokens,
                labels,
                segs,
                clss,
                meta_field,
                disco_label,
                disco_span
                ):
        if detect_nan(tokens['bert']) or detect_nan(labels) or detect_nan(segs) or detect_nan(clss):
            print("NAN")
            exit()
        # print(meta_field)
        with autograd.detect_anomaly():
            input_ids = tokens[self._index]
            input_mask = (input_ids != 0).long()
            segs = segs.long()
            encoded_layers, _ = self.bert_model(input_ids=input_ids,
                                                token_type_ids=segs,
                                                attention_mask=input_mask)
            # print("raodmark2")
            top_vec = encoded_layers[-1]
            # top_vec = self._dropout(top_vec)
            if self._use_disco:
                attended_text_embeddings = self._span_extractor.forward(top_vec, disco_span,input_mask)
                pass
            else:
                sent_rep, sent_mask = efficient_head_selection(top_vec, clss)
            # sent_rep: batch size, sent num, bert hid dim
            # sent_mask: batch size, sent num

            # sent_rep = self._layer_norm(sent_rep, sent_mask)
            sent_rep = self._graph_encoder.transform_sent_rep(sent_rep, sent_mask)
            # sent_rep = self._layer_norm(sent_rep, sent_mask)
            batch_size, sent_num = sent_mask.shape
            scores = self._sigmoid(self._classification_layer(self._dropout(sent_rep)))
            scores = scores.squeeze(-1)
            # scores = scores + (sent_mask.float() - 1)
            # logits = self._transfrom_layer(logits)
            # probs = torch.nn.functional.softmax(logits, dim=-1)

            output_dict = {"scores": scores,
                           # "probs": probs,
                           'mask': sent_mask,
                           "meta": meta_field}

            if labels is not None:
                # logits: batch size, sent num
                # labels: batch size, sent num
                # sent_mask: batch size, sent num
                # flatten_scores = scores.view(batch_size * sent_num)
                # flatten_labels = labels.view(batch_size * sent_num).float()
                # print(scores)
                # print(labels)
                labels = torch.nn.functional.relu(labels)
                raw_loss = self._loss(scores, labels.float())
                # print(loss.shape)
                # print(sent_mask.shape)
                # print(loss)

                loss = raw_loss * sent_mask.float()
                if random.random() < 0.01:
                    print(loss.data[0])
                loss = torch.sum(loss)
                output_dict["loss"] = loss
                # self._accuracy(flatten_logits, flatten_labels, mask=sent_mask.view(-1))
            # print(output_dict)

            ##
            # for name, param in self.named_parameters():
            #     print(name)
            # print(param)
            # if param.grad is not None:
            # print(param.grad[0])
            ##
            type = meta_field[0]['type']
            if type == 'valid' or type == 'test':
                output_dict = self.decode(output_dict)
            return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        # probs: batch size, sent num, 2
        # masks: batch size, sent num [binary]
        scores = output_dict['scores']
        masks = output_dict['mask']
        meta = output_dict['meta']
        # expanded_msk = masks.unsqueeze(2).expand_as(probs)
        # expanded_msk = masks.unsqueeze(2)
        tuned_probs = scores + (masks.float() - 1) * 10
        # scores = scores + (sent_mask.float() - 1)
        # tuned_probs = tuned_probs.cpu().data.numpy()[:, :, 1]
        tuned_probs = tuned_probs.cpu().data.numpy()

        batch_size, sent_num = masks.shape

        for b in range(batch_size):
            this_meta = meta[b]
            src = this_meta['src_txt']
            tgt = this_meta['tgt_txt']
            tgt_as_list = tgt.split('<q>')
            this_prob = tuned_probs[b]
            sel_indexs = np.argsort(-this_prob)

            trigrams = set()
            predictions = []
            predictions_idx = []
            for sel_i in sel_indexs:
                try:
                    candidate = src[sel_i]
                    cand_trigram = extract_n_grams(candidate)
                    if trigrams.isdisjoint(cand_trigram):
                        predictions.append(candidate)
                        predictions_idx.append(sel_i)
                        trigrams.update(cand_trigram)
                except IndexError:
                    logger.warning("Index Error\n{}".format(this_meta['src_txt']))
                    # print("Index Error\n{}\n{}".format(this_meta['src_txt'], sel_indexs))
                    continue
                if len(predictions) >= self._pred_length:
                    break
            # if random.random() < 0.001:
            #     print("\n".join(predictions))
            #     print(predictions_idx)
            # print(tgt_as_list)
            self._rouge(pred="<q>".join(predictions), ref=tgt)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        dict_of_rouge = self._rouge.get_metric(reset)
        metrics = {
            # 'accuracy': self._accuracy.get_metric(reset),
            'R_1': dict_of_rouge[self._rouge.name + '_1'],
            'R_2': dict_of_rouge[self._rouge.name + '_2'],
            'R_L': dict_of_rouge[self._rouge.name + '_L']
        }
        return metrics


# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import os
import allennlp


def build_vocab():
    from allennlp.commands.make_vocab import make_vocab_from_params

    jsonnet_file = os.path.join(root, 'configs/baseline_bert.jsonnet')
    params = Params.from_file(jsonnet_file)
    make_vocab_from_params(params, '/datadrive/bert_vocab')


if __name__ == '__main__':
    if os.path.isdir('/datadrive'):
        root = "/datadrive/GETSum/"
    elif os.path.isdir('/scratch/cluster/jcxu'):
        root = '/scratch/cluster/jcxu/GETSum'
    else:
        raise NotImplementedError
    logger.info("AllenNLP version {}".format(allennlp.__version__))

    jsonnet_file = os.path.join(root, 'configs/baseline_bert.jsonnet')
    params = Params.from_file(jsonnet_file)
    print(params.params)
    serialization_dir = tempfile.mkdtemp(prefix=os.path.join(root, 'tmp_exps'))
    model = train_model(params, serialization_dir)
    # serialization_dir = '/backup3/jcxu/NeuSegSum/tmp_expsn7oe84zt'
    # print(serialization_dir)
    # params = Params.from_file(jsonnet_file)
    print(serialization_dir)
