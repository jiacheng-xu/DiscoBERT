import logging, itertools, dgl, random, torch, tempfile
import multiprocessing
from typing import Dict, List, Optional, Union
import numpy as np
from allennlp.commands.fine_tune import fine_tune_model_from_file_paths
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel, PretrainedBertEmbedder
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel

from model.archival_gnns import GraphEncoder
from model.decoding_util import decode_entrance

flatten = lambda l: [item for sublist in l for item in sublist]
from torch.nn.functional import nll_loss
import torch.nn as nn
from torch import autograd
from allennlp.common import FromParams
from model.gcn import GCN_layers
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common import Registrable
from model.data_reader import CNNDMDatasetReader
# from model.pythonrouge_metrics import RougeStrEvaluation
from model.pyrouge_metrics import PyrougeEvaluation
from multiprocessing import Pool, TimeoutError
import time
import os

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from pytorch_pretrained_bert import BertConfig

from model.model_util import *


def run_eval_worker(obj):
    return obj.get_metric(True)


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
        # self.ln = MaskedLayerNorm(size=768)

    def transform_sent_rep(self, sent_rep, sent_mask):
        # return self.ln.forward(sent_rep, sent_mask)
        return sent_rep


from collections import deque

from allennlp.modules.feedforward import FeedForward

from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.span_extractors.span_extractor import SpanExtractor

from model.model_util import easy_post_processing


@Model.register("tensor_bert")
class TensorBertSum(Model):
    def __init__(self, vocab: Vocabulary,
                 # bert_model: Union[str, BertModel],
                 # bert_config_file: str,
                 debug: bool,
                 bert_max_length: int,
                 multi_orac: bool,
                 graph_encoder: GraphEncoder,
                 span_extractor: SpanExtractor,
                 trainable: bool = True,
                 use_disco: bool = True,
                 use_disco_graph=True,
                 use_coref: bool = False,
                 index: str = "bert",
                 dropout: float = 0.2,
                 tmp_dir: str = '/datadrive/tmp/',
                 stop_by_word_count: bool = True,
                 use_pivot_decode: bool = False,
                 trigram_block=True,
                 min_pred_word: int = 30,
                 max_pred_word: int = 80,
                 step: int = 10,
                 min_pred_unit: int = 6,
                 max_pred_unit: int = 9,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        # super(TensorBertSum, self).__init__(vocab, regularizer)
        super(TensorBertSum, self).__init__(vocab)
        self.debug = debug
        self.embedder = PretrainedBertEmbedder('bert-base-uncased', requires_grad=True, top_layer_only=True)

        if bert_max_length > 512:
            first_half = self.embedder.bert_model.embeddings.position_embeddings.weight
            # ts = torch.zeros_like(first_half, dtype=torch.float32)
            # second_half = ts.new_tensor(first_half, requires_grad=True)

            second_half = torch.zeros_like(first_half, dtype=torch.float32, requires_grad=True)

            # second_half = torch.empty(first_half.size(), dtype=torch.float32,requires_grad=True)
            # torch.nn.init.normal_(second_half, mean=0.0, std=1.0)
            # print('normal')
            out = torch.cat([first_half, second_half], dim=0)
            # print('-- copy')
            self.embedder.bert_model.embeddings.position_embeddings.weight = torch.nn.Parameter(out)
            self.embedder.bert_model.embeddings.position_embeddings.num_embeddings = 512 * 2
            self.embedder.max_pieces = 512 * 2

        """
        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model
    
        if bert_max_length > 512:
            first_half = self.bert_model.embeddings.position_embeddings.weight
            ts = torch.zeros_like(first_half, dtype=torch.float32)
            second_half = ts.new_tensor(first_half, requires_grad=True)
            # print(out.size())
            # exit()
            # out = torch.nn.functional.interpolate(torch.unsqueeze(out.permute((1, 0), 0), 0), size=bert_max_length,
            #                                       mode='linear')
            # out = out.squeeze(0)
            # out = out.permute((1, 0))
            out = torch.cat([first_half, second_half], dim=0)
            self.bert_model.embeddings.position_embeddings.weight = torch.nn.Parameter(out)
            self.bert_model.embeddings.position_embeddings.num_embeddings = 512*2

        self.bert_model.config = BertConfig.from_json_file(bert_config_file)
        
        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        in_features = self.bert_model.config.hidden_size
        """

        self._graph_encoder = graph_encoder
        in_features = 768
        self._index = index
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(in_features, 1)
        # self._transfrom_layer = torch.nn.Linear(100, 2)
        # self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.BCELoss(reduction='none')
        self._layer_norm = MaskedLayerNorm(768)
        self._multi_orac = multi_orac

        # ROUGES
        self._stop_by_word_count = stop_by_word_count
        if stop_by_word_count:
            self.slot_num = int((max_pred_word - min_pred_word) / step)
            for i in range(self.slot_num):
                setattr(self, "rouge_{}".format(i),
                        PyrougeEvaluation(name='rouge_{}'.format(i), cand_path=tmp_dir, ref_path=tmp_dir,
                                          path_to_valid=tmp_dir))
        else:
            self._min_pred_unit = min_pred_unit
            self._max_pred_unit = max_pred_unit
            for i in range(min_pred_unit, max_pred_unit):
                setattr(self, "rouge_{}".format(i),
                        PyrougeEvaluation(name='rouge_{}'.format(i), cand_path=tmp_dir, ref_path=tmp_dir,
                                          path_to_valid=tmp_dir))
        self._sigmoid = nn.Sigmoid()
        initializer(self._classification_layer)

        self._use_disco = use_disco

        self._use_disco_graph = use_disco_graph
        self._span_extractor = span_extractor
        self._use_coref = use_coref

        self._trigram_block = trigram_block
        self._use_pivot_decode = use_pivot_decode
        self._min_pred_word = min_pred_word
        self._max_pred_word = max_pred_word
        self._step = step

    """
    def tear_apart_input_for_bert(self, inps, segs, stride=100):
        # if inps.size(-1) % 2 != 0:
        #     pass
        # print(inps.size())
        first_inps = inps[:, :512]
        second_inps = inps[:, 512 - stride:]
        first_segs = segs[:, :512]
        second_segs = segs[:, 512 - stride:]
        first_output = self.embedder.forward(first_inps, first_segs)
        sec_output = self.embedder.forward(second_inps, second_segs)[:, stride:, :]
        # print(first_output.size())
        # print(sec_output.size())
        cat = torch.cat((first_output, sec_output), dim=1)
        # print(cat.size())
        return first_output
    """

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
                disco_span,
                # disco_coref_graph,
                # disco_rst_graph
                ):
        if detect_nan(tokens['bert']) or detect_nan(labels) or detect_nan(segs) or detect_nan(clss):
            print("NAN")
            exit()
        # print(meta_field)
        # if random.random() < 0.001:
        #     print('real zero')
        with autograd.detect_anomaly():
            input_ids = tokens[self._index]
            ## old bert
            input_mask = (input_ids != 0).long()
            # segs = segs.long()
            # encoded_layers, _ = self.bert_model(input_ids=input_ids,
            #                                     token_type_ids=segs,
            #                                     attention_mask=input_mask)
            ####### end of old bert
            # start of new bert
            output = self.embedder.forward(input_ids=input_ids,
                                           token_type_ids=segs
                                           )
            # print(output.size())
            top_vec = output
            # hardcode bert
            # top_vec = self.tear_apart_input_for_bert(input_ids, segs)
            # print(output.size())
            # exit()
            # print("raodmark2")
            # top_vec = encoded_layers[-1]
            # top_vec = self._dropout(top_vec)
            label_to_use = None
            if self._use_disco:
                # if self._use_disco:
                disco_mask = (disco_span[:, :, 0] >= 0).long()
                disco_span = torch.nn.functional.relu(disco_span.float()).long()
                attended_text_embeddings = self._span_extractor.forward(top_vec, disco_span, input_mask, disco_mask)
                encoder_output, encoder_output_msk = attended_text_embeddings, disco_mask
                label_to_use = disco_label

            else:
                sent_rep, sent_mask = efficient_head_selection(top_vec, clss)
                # sent_rep: batch size, sent num, bert hid dim
                # sent_mask: batch size, sent num
                encoder_output, encoder_output_msk = sent_rep, sent_mask
                label_to_use = labels

            # sent_rep = self._layer_norm(sent_rep, sent_mask)
            if self._use_disco_graph:
                encoder_output_af_graph = self._graph_encoder.transform_sent_rep(encoder_output,
                                                                                 encoder_output_msk,
                                                                                 meta_field, 'disco_rst_graph')
            elif self._use_coref:
                encoder_output_af_graph = self._graph_encoder.transform_sent_rep(encoder_output, encoder_output_msk,meta_field,
                                                                                 'disco_coref_graph')
            else:
                encoder_output_af_graph = encoder_output
            # sent_rep = self._layer_norm(sent_rep, sent_mask)

            # batch_size, sent_num = sent_mask.shape

            scores = self._sigmoid(self._classification_layer(self._dropout(encoder_output_af_graph)))
            scores = scores.squeeze(-1)
            # scores = scores + (sent_mask.float() - 1)
            # logits = self._transfrom_layer(logits)
            # probs = torch.nn.functional.softmax(logits, dim=-1)

            output_dict = {"scores": scores,
                           # "probs": probs,
                           "mask": encoder_output_msk,
                           # 'mask': sent_mask,
                           # 'disco_mask': disco_mask,
                           "meta": meta_field}

            if label_to_use is not None:
                # logits: batch size, sent num
                # labels: batch size, oracle_num, sent num
                # sent_mask: batch size, sent num
                # flatten_scores = scores.view(batch_size * sent_num)
                # flatten_labels = labels.view(batch_size * sent_num).float()
                # print(scores)
                # print(labels)
                if self._multi_orac:
                    seq_len = scores.size()[-1]
                    scores = scores.unsqueeze(1)
                    encoder_output_msk = encoder_output_msk.unsqueeze(1)
                    scores = scores.expand_as(label_to_use).contiguous().view(-1, seq_len)
                    encoder_output_msk = encoder_output_msk.expand_as(label_to_use).contiguous().view(-1, seq_len)
                    label_to_use = label_to_use.view(-1, seq_len)
                else:
                    label_to_use = label_to_use[:, 0, :]
                label_to_use = torch.nn.functional.relu(label_to_use)
                raw_loss = self._loss(scores, label_to_use.float())
                # print(loss.shape)
                # print(sent_mask.shape)
                # print(loss)

                loss = raw_loss * encoder_output_msk.float()
                if random.random() < 0.0001:
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
            # print(len(self.rouge_0.pred_str_bag))
            if (type == 'valid' or type == 'test') or (self.debug):
                output_dict = self.decode(output_dict, trigram_block=self._trigram_block,
                                          stop_by_word_cnt=self._stop_by_word_count)
            return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor],
               trigram_block: bool = True,
               use_pivot_decode: bool = False,
               stop_by_word_cnt: bool = False
               ):
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
            pred_word_list_strs, tgt_str = decode_entrance(tuned_probs[b], meta[b], self._use_disco,
                                                           trigram_block,
                                                           use_pivot_decode, stop_by_word_cnt,
                                                           self._min_pred_word,
                                                           self._max_pred_word,
                                                           self._step,
                                                           self._min_pred_unit,
                                                           self._max_pred_unit
                                                           )
            if self._stop_by_word_count:
                for l in range(self.slot_num):
                    getattr(self, 'rouge_{}'.format(l))(pred="<q>".join(pred_word_list_strs[l]),
                                                        ref=tgt_str)
            else:
                for l in range(self._min_pred_unit, self._max_pred_unit):
                    getattr(self, 'rouge_{}'.format(l))(pred="<q>".join(pred_word_list_strs[l]),
                                                        ref=tgt_str)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # dict_of_rouge = self._rouge.get_metric(reset)
        dict_of_rouge = {}
        # print(reset)
        if reset:
            if self._stop_by_word_count:
                obj_list = [(getattr(self, 'rouge_{}'.format(l)),) for l in range(self.slot_num)]
            else:
                obj_list = [(getattr(self, 'rouge_{}'.format(l)),) for l in
                            range(self._min_pred_unit, self._max_pred_unit)]

            pool = multiprocessing.Pool(processes=10)
            results = pool.starmap(run_eval_worker, obj_list)

            if self._stop_by_word_count:

                for l in range(self.slot_num):
                    getattr(self, 'rouge_{}'.format(l)).reset()
            else:
                for l in range(self._min_pred_unit, self._max_pred_unit):
                    getattr(self, 'rouge_{}'.format(l)).reset()

            for r in results:
                dict_of_rouge = {**dict_of_rouge, **r}
        else:
            if self._stop_by_word_count:
                for l in range(self.slot_num):
                    _d = getattr(self, 'rouge_{}'.format(l)).get_metric(reset)
                    dict_of_rouge = {**dict_of_rouge, **_d}
            else:
                for l in range(self._min_pred_unit, self._max_pred_unit):
                    _d = getattr(self, 'rouge_{}'.format(l)).get_metric(reset)
                    dict_of_rouge = {**dict_of_rouge, **_d}

        # find best f1
        best_key, best_val = "", -1
        # print(dict_of_rouge)
        for key, val in dict_of_rouge.items():
            if key.endswith("_1"):
                if val > best_val:
                    best_val = val
                    best_key = key
        best_name = best_key[:-2]
        metrics = {
            # 'accuracy': self._accuracy.get_metric(reset),
            'R_1': dict_of_rouge['{}_1'.format(best_name)],
            'R_2': dict_of_rouge['{}_2'.format(best_name)],
            'R_L': dict_of_rouge['{}_L'.format(best_name)]
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

    finetune = True

    logger.info("AllenNLP version {}".format(allennlp.__version__))

    jsonnet_file = os.path.join(root, 'configs/baseline_bert.jsonnet')
    params = Params.from_file(jsonnet_file)
    print(params.params)

    serialization_dir = tempfile.mkdtemp(prefix=os.path.join(root, 'tmp_exps'))
    if finetune:
        model_arch = '/datadrive/GETSum/tmp_exps8b8br1d3'
        fine_tune_model_from_file_paths(model_arch,
                                        os.path.join(root, 'configs/baseline_bert.jsonnet'),
                                        # os.path.join(root, 'configs/finetune.jsonnet'),
                                        serialization_dir
                                        )
    else:
        model = train_model(params, serialization_dir)
    print(serialization_dir)
