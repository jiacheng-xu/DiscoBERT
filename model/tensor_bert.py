import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.nn.functional import nll_loss
from overrides import overrides
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
import random
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.commands.train import train_model

from allennlp.common import Params
from allennlp.data.iterators import DataIterator
# from segsum.dataset_readers.seg_sum_read import SegSumDatasetReader
from model.data_reader import CNNDMDatasetReader
from allennlp.commands.evaluate import evaluate
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.archival import load_archive

# from model.pythonrouge_metrics import RougeStrEvaluation
from model.pyrouge_metrics import PyrougeEvaluation

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import tempfile
from pytorch_pretrained_bert import BertConfig


def detect_nan(input_tensor) -> bool:
    if torch.sum(torch.isnan(input_tensor)) > 0:
        return True
    else:
        return False


def flatten_2d_matrix_to_1d(two_dim_matrix, word_num):
    batch_size, sent_num = two_dim_matrix.shape
    bias = torch.arange(start=0, end=batch_size, dtype=torch.long, device=two_dim_matrix.device,
                        requires_grad=False) * word_num
    bias = bias.view(-1, 1)
    bias = bias.repeat(1, sent_num).view(-1)
    flatten_2d_raw = two_dim_matrix.view(-1)

    return (flatten_2d_raw + bias).long()


def flatten_3d_tensor_to_2d(three_dim_tensor):
    # flatten the first two dim
    shape0, shape1, shape2 = three_dim_tensor.shape
    return three_dim_tensor.view(shape0 * shape1, shape2)


def efficient_head_selection(top_vec, clss):
    assert top_vec.shape[0] == clss.shape[0]
    word_num = top_vec.shape[1]
    batch_size = top_vec.shape[0]
    sent_num = clss.shape[1]
    sent_mask = (clss >= -0.0001).float()  # batch size, max sent num
    # if random.random()<0.01:
    #     print(sent_mask)
    clss_non_neg = torch.nn.functional.relu(clss).long()

    matrix_top_vec = flatten_3d_tensor_to_2d(top_vec)  # batch size, word seq len, hdim
    vec_clss_non_neg = flatten_2d_matrix_to_1d(clss_non_neg, word_num)
    flatten_selected_sent_rep = torch.index_select(matrix_top_vec, 0, vec_clss_non_neg)

    selected_sent_rep = flatten_selected_sent_rep.view(batch_size, sent_num, -1)
    selected_sent_rep = selected_sent_rep * sent_mask.unsqueeze(-1)
    # print(selected_sent_rep.shape)
    # print(sent_mask.shape)
    return selected_sent_rep, sent_mask


def extract_n_grams(inp_str, ngram: int = 3, connect_punc='_') -> set:
    inp_list = inp_str.split(" ")
    if len(inp_list) < 3:
        return set()
    tmp = []
    for idx in range(len(inp_list) - ngram + 1):
        this = [inp_list[idx + j] for j in range(ngram)]
        tmp.append(connect_punc.join(this))
    return set(tmp)


import torch.nn as nn

from allennlp.training.metrics import Metric
from torch import autograd

from torch.nn.init import xavier_uniform_


@Model.register("tensor_bert")
class TensorBertSum(Model):
    def __init__(self, vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 trainable: bool = True,
                 index: str = "bert",
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        # super(TensorBertSum, self).__init__(vocab, regularizer)
        super(TensorBertSum, self).__init__(vocab)
        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        self.bert_model.config = BertConfig.from_json_file("/datadrive/msSum/configs/BertSumConfig.json")

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        in_features = self.bert_model.config.hidden_size
        self._index = index
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(in_features, 1)
        # self._transfrom_layer = torch.nn.Linear(100, 2)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.BCELoss(reduction='none')
        self._rouge = PyrougeEvaluation(name='rouge')
        self._sigmoid = nn.Sigmoid()
        self.bert_model.requires_grad = False
        initializer(self._classification_layer)
        # xavier_uniform_(self._classification_layer._parameters())

    @overrides
    def forward(self, tokens,
                labels,
                segs,
                clss,
                meta_field
                # tokens: Dict[str, torch.LongTensor]
                ):
        # for m in meta_field:
        #     print(m)
        # print(clss[1])
        # print(labels[1])
        # print(tokens['bert'][1].tolist())
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
            sent_rep, sent_mask = efficient_head_selection(top_vec, clss)
            # sent_rep: batch size, sent num, bert hid dim
            # sent_mask: batch size, sent num
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
                    print(loss)
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
        tuned_probs = scores + (masks.float() - 1)
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
                    print("Index Error")
                    break
                if len(predictions) >= 3:
                    break
            # if random.random() < 0.001:
            #     print("\n".join(predictions))
            #     print(predictions_idx)
            # print(tgt_as_list)
            self._rouge(pred="<q>".join(predictions), ref=tgt)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        dict_of_rouge = self._rouge.get_metric(reset)
        metrics = {'accuracy': self._accuracy.get_metric(reset),
                   'R_1': dict_of_rouge[self._rouge.name + '_1'],
                   'R_2': dict_of_rouge[self._rouge.name + '_2'],
                   'R_L': dict_of_rouge[self._rouge.name + '_L']
                   }
        return metrics


import logging
import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import os
import allennlp


def build_vocab():
    from allennlp.commands.make_vocab import make_vocab_from_params

    jsonnet_file = os.path.join(root, 'configs/baseline_bert.jsonnet')
    params = Params.from_file(jsonnet_file)
    make_vocab_from_params(params, '/datadrive/bert_vocab')


if __name__ == '__main__':
    root = "/datadrive/msSum/"
    print(allennlp.__version__)
    # build_vocab()
    # exit()
    # print(extract_n_grams("aaaa bbbbb cccc ddd ee ff"))
    # {'aaaa_bbbbb_cccc', 'cccc_ddd_ee', 'ddd_ee_ff', 'bbbbb_cccc_ddd'}

    jsonnet_file = os.path.join(root, 'configs/baseline_bert.jsonnet')
    params = Params.from_file(jsonnet_file)

    serialization_dir = tempfile.mkdtemp(prefix=os.path.join(root, 'tmp_exps'))
    model = train_model(params, serialization_dir)
    # serialization_dir = '/backup3/jcxu/NeuSegSum/tmp_expsn7oe84zt'
    # print(serialization_dir)
    params = Params.from_file(jsonnet_file)
