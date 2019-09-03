import logging, itertools, dgl, random, torch, tempfile
import multiprocessing
from typing import Dict, List, Optional, Union
import numpy as np
from allennlp.commands.fine_tune import fine_tune_model_from_file_paths
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel, PretrainedBertEmbedder
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel
from collections import OrderedDict
from model.archival_gnns import GraphEncoder
from model.decoding_util import decode_entrance
from model.sem_red_map import MapKiosk

torch.multiprocessing.set_sharing_strategy('file_system')

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
                 bert_pretrain_model: str,
                 bert_max_length: int,
                 multi_orac: bool,

                 semantic_red_map: bool,  # use redundancy map or not
                 semantic_red_map_key: str,  # p or f
                 semantic_red_map_loss: str,  # bin or mag

                 pair_oracle: bool,  # use pairwise estimation as salience estimation
                 fusion_feedforward: FeedForward,
                 semantic_feedforard: FeedForward,
                 graph_encoder: GraphEncoder,
                 span_extractor: SpanExtractor,
                 matrix_attn: MatrixAttention,
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
                 threshold_red_map: List = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        # super(TensorBertSum, self).__init__(vocab, regularizer)
        super(TensorBertSum, self).__init__(vocab)
        self.debug = debug
        self.embedder = PretrainedBertEmbedder('bert-base-uncased', requires_grad=True, top_layer_only=True)

        self.bert_pretrain_model = bert_pretrain_model

        if bert_max_length > 512:
            first_half = self.embedder.bert_model.embeddings.position_embeddings.weight
            # ts = torch.zeros_like(first_half, dtype=torch.float32)
            # second_half = ts.new_tensor(first_half, requires_grad=True)

            second_half = torch.zeros_like(first_half, dtype=torch.float32, requires_grad=True)

            # second_half = torch.empty(first_half.size(), dtype=torch.float32,requires_grad=True)
            # torch.nn.init.normal_(second_half, mean=0.0, std=1.0)
            out = torch.cat([first_half, second_half], dim=0)
            self.embedder.bert_model.embeddings.position_embeddings.weight = torch.nn.Parameter(out)
            self.embedder.bert_model.embeddings.position_embeddings.num_embeddings = 512 * 2
            self.embedder.max_pieces = 512 * 2
        if bert_pretrain_model is not None:
            model_dump: OrderedDict = torch.load(os.path.join(bert_pretrain_model, 'best.th'))
            trimmed_dump_embedder = OrderedDict()
            for k, v in model_dump.items():
                if k.startswith("embedder"):
                    trimmed_dump_embedder[k] = v
            self.load_state_dict(trimmed_dump_embedder)
            print('finish loading pretrained bert')

        in_features = 768
        self._index = index
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(in_features, 1)
        self._loss = torch.nn.BCELoss(reduction='none')
        # self._loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._layer_norm = MaskedLayerNorm(768)
        self._multi_orac = multi_orac

        # ROUGES
        self._stop_by_word_count = stop_by_word_count
        self._threshold_red_map = threshold_red_map
        if stop_by_word_count:
            self.slot_num = int((max_pred_word - min_pred_word) / step)
            for i in range(self.slot_num):
                setattr(self, "rouge_{}".format(i),
                        PyrougeEvaluation(name='rouge_{}'.format(i), cand_path=tmp_dir, ref_path=tmp_dir,
                                          path_to_valid=tmp_dir)
                        )
        else:
            self._min_pred_unit = min_pred_unit
            self._max_pred_unit = max_pred_unit
            for i in range(min_pred_unit, max_pred_unit):
                for ths in threshold_red_map:
                    setattr(self, "rouge_{}_{}".format(i, ths),
                            PyrougeEvaluation(name="rouge_{}_{}".format(i, ths),
                                              cand_path=tmp_dir, ref_path=tmp_dir,
                                              path_to_valid=tmp_dir))
        self._sigmoid = nn.Sigmoid()
        initializer(self._classification_layer)

        self._use_disco = use_disco

        self._use_disco_graph = use_disco_graph
        if use_disco_graph:
            self.disco_graph_encoder = graph_encoder
        self._use_coref = use_coref
        if use_coref:
            self.coref_graph_encoder = graph_encoder
        if self._use_coref and self._use_disco_graph:
            self._fusion_feedforward = fusion_feedforward
        self._span_extractor = span_extractor

        self._trigram_block = trigram_block
        self._use_pivot_decode = use_pivot_decode
        self._min_pred_word = min_pred_word
        self._max_pred_word = max_pred_word
        self._step = step

        self._semantic_red_map = semantic_red_map
        self._semantic_red_map_loss = semantic_red_map_loss
        self._semantic_red_map_key = semantic_red_map_key

        if self._semantic_red_map:
            self.red_matrix_attn = matrix_attn
            self._semantic_feedforard = semantic_feedforard
        self._pair_oracle = pair_oracle
        if self._pair_oracle:
            self.pair_matrix_attn = matrix_attn

    def transform_sent_rep(self, sent_rep, sent_mask):
        init_graphs = self._graph_encoder.convert_sent_tensors_to_graphs(sent_rep, sent_mask)
        unpadated_graphs = []
        for g in init_graphs:
            updated_graph = self._graph_encoder(g)
            unpadated_graphs.append(updated_graph)
        recovered_sent = torch.stack(unpadated_graphs, dim=0)
        assert recovered_sent.shape == sent_rep.shape
        return recovered_sent

    def compute_standard_loss(self, encoder_output_af_graph,
                              encoder_output_msk,
                              meta_field,
                              label_to_use
                              ):

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
                       "meta": meta_field
                       }

        if label_to_use is not None:
            # logits: batch size, sent num
            # labels: batch size, oracle_num, sent num
            # sent_mask: batch size, sent num
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

        return output_dict

    def compute_pair_loss(self):
        pass

    @overrides
    def forward(self, tokens,
                labels,
                segs,
                clss,
                meta_field,
                disco_label,
                disco_span,
                **kwargs
                # unigram_overlap,
                # red_map_p_mask,
                # red_map_p_opt_idx
                ):
        with autograd.detect_anomaly():
            input_ids = tokens[self._index]
            input_mask = (input_ids != 0).long()
            output = self.embedder.forward(input_ids=input_ids,
                                           token_type_ids=segs
                                           )
            top_vec = output
            # hardcode bert
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

            if self._use_disco_graph and not self._use_coref:
                encoder_output_af_graph = self.disco_graph_encoder.transform_sent_rep(encoder_output,
                                                                                      encoder_output_msk,
                                                                                      meta_field, 'disco_rst_graph')
            elif self._use_coref and not self._use_disco_graph:
                encoder_output_af_graph = self.coref_graph_encoder.transform_sent_rep(encoder_output,
                                                                                      encoder_output_msk,
                                                                                      meta_field,
                                                                                      'disco_coref_graph')
            elif self._use_coref and self._use_disco_graph:
                encoder_output_disco = self.disco_graph_encoder.transform_sent_rep(encoder_output,
                                                                                   encoder_output_msk,
                                                                                   meta_field, 'disco_rst_graph')
                encoder_output_coref = self.coref_graph_encoder.transform_sent_rep(encoder_output, encoder_output_msk,
                                                                                   meta_field,
                                                                                   'disco_coref_graph')
                encoder_output_combine = torch.cat([encoder_output_coref, encoder_output_disco], dim=2)
                encoder_output_af_graph = self._fusion_feedforward.forward(encoder_output_combine)
            else:
                encoder_output_af_graph = encoder_output

            if self._pair_oracle:
                # TODO

                self.compute_pair_loss(encoder_output_af_graph,
                                       encoder_output_msk)
                raise NotImplementedError
            else:
                output_dict = self.compute_standard_loss(encoder_output_af_graph,
                                                         encoder_output_msk,
                                                         meta_field,
                                                         label_to_use)

            # Do we need to train an explict redundancy model?
            if self._semantic_red_map:
                encoder_output_af_graph = self._layer_norm.forward(encoder_output_af_graph, encoder_output_msk)
                encoder_output_af_graph = self._semantic_feedforard.forward(encoder_output_af_graph)
                encoder_output_af_graph = self._layer_norm.forward(encoder_output_af_graph, encoder_output_msk)
                attn_feat = self.red_matrix_attn.forward(encoder_output_af_graph, encoder_output_af_graph)

                attn_feat = torch.nn.functional.sigmoid(attn_feat)
                batch_size = attn_feat.shape[0]
                valid_len = attn_feat.shape[1]
                # red_map_p_mask: batch, len, len
                # red_map_p_opt_idx: batch, len
                red_p_pos = kwargs['red_{}_pos'.format(self._semantic_red_map_key)]
                red_p_neg = kwargs['red_{}_neg'.format(self._semantic_red_map_key)]

                if self._semantic_red_map_loss == 'bin':
                    training_mask = red_p_pos + red_p_neg  # these are the mask for trainign bit
                    training_mask = torch.nn.functional.relu(training_mask)
                    red_p_pos = torch.nn.functional.relu(red_p_pos)
                    # red_p_neg = torch.nn.functional.relu(red_p_neg).byte()

                    # pos_feat = torch.masked_select(attn_feat, mask=red_p_pos)
                    # neg_feat = torch.masked_select(attn_feat, mask=red_p_neg)
                    red_loss = self._loss(attn_feat, red_p_pos)
                    # red_loss = torch.sum(neg_feat) - torch.sum(pos_feat)
                    red_loss = red_loss * training_mask
                    red_loss = torch.sum(red_loss) / 50
                    if random.random() < 0.02:
                        print("margin loss: {}".format(red_loss))
                    output_dict["loss"] += red_loss
                elif self._semantic_red_map_loss == 'mag':
                    red_map_p_mask = kwargs['red_map_p_supervision_mask']
                    red_map_p_opt_idx = kwargs['red_map_p_opt_idx']
                    unigram_overlap = kwargs['unigram_overlap']
                    red_map_p_mask = red_map_p_mask.float()
                    rt_sel = efficient_oracle_selection(attn_feat,
                                                        red_map_p_opt_idx
                                                        )
                    # red_map_p_opt_idx_mask = red_map_p_opt_idx_mask.unsqueeze(2).expand_as(red_map_p_mask).float()
                    # two masks to use
                    objective = ((
                                         0.5 - unigram_overlap) - rt_sel + attn_feat) * red_map_p_mask  # * red_map_p_opt_idx_mask
                    margin_loss = torch.nn.functional.relu(objective)


                else:
                    pass

                output_dict['scores_matrix_red'] = attn_feat

                if random.random() < 0.008:
                    # print(margin_loss.data[0])
                    print(attn_feat.data.cpu()[0][0])
                    # print(attn_feat.data.cpu()[0][-1])

                # scores = self._sigmoid(self._classification_layer(self._dropout(
                #     self._univec_feedforward.forward(encoder_output_af_graph))))  # batch, sent_num, 1
                # scores = scores.squeeze(-1)
                #
                # diag_scores = torch.diag_embed(scores)
                # diag_mask = (diag_scores > 0).float()
                #
                # scores_matrix = self.matrix_attn.forward(encoder_output_af_graph,
                #                                          encoder_output_af_graph)  # batch, sent_num, sent_num
                # fill the diag of scores matrix with the uni scores
                # the salience map's diag is R[x], all other cells are R[x +y]
                # scores_matrix = diag_mask * diag_scores + (1 - diag_mask) * scores_matrix

                # label_map = locals()[self._semantic_red_map_key]
                # label_map_mask = label_map >= 0
                # label_map = torch.nn.functional.relu(label_map)

            type = meta_field[0]['type']
            source_name = meta_field[0]['source']
            # print(len(self.rouge_0.pred_str_bag))
            if (type == 'valid' or type == 'test') or (self.debug):
                # for ths in self._threshold_red_map:
                output_dict = self.decode(output_dict,
                                          trigram_block=self._trigram_block,
                                          min_pred_unit=self._min_pred_unit,
                                          max_pred_unit=self._max_pred_unit,
                                          sem_red_matrix=self._semantic_red_map,
                                          pair_oracle=self._pair_oracle,
                                          stop_by_word_cnt=self._stop_by_word_count,
                                          threshold_for_red_map=0,
                                          source_name=source_name
                                          )
            return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor],
               trigram_block: bool = True,
               min_pred_unit: int = 4,
               max_pred_unit: int = 7,
               sem_red_matrix: bool = False,
               pair_oracle: bool = False,
               stop_by_word_cnt: bool = False,
               threshold_for_red_map: float = 0.05,
               source_name: str = 'dailymail'
               ):
        # probs: batch size, sent num, 2
        # masks: batch size, sent num [binary]

        masks = output_dict['mask']
        meta = output_dict['meta']
        batch_size = len(meta)
        # expanded_msk = masks.unsqueeze(2).expand_as(probs)
        # expanded_msk = masks.unsqueeze(2)

        if pair_oracle:
            raise NotImplementedError
            pass
        else:
            scores = output_dict['scores']
            tuned_probs = scores + (masks.float() - 1) * 10
            tuned_probs = tuned_probs.cpu().data.numpy()
        if sem_red_matrix:
            scores_matrix = output_dict['scores_matrix_red']
            masks_matrix = masks.unsqueeze(2).expand_as(scores_matrix)
            rot_masks = torch.rot90(masks_matrix, 1, [1, 2])
            _mask_matrix = rot_masks * masks_matrix
            tuned_scores_matrix = scores_matrix + (_mask_matrix.float() - 1) * 10
            tuned_mat_probs = tuned_scores_matrix.cpu().data.numpy()
        else:
            tuned_mat_probs = [None for _ in range(batch_size)]

        batch_size, sent_num = masks.shape
        for b in range(batch_size):
            doc_id = meta[b]['doc_id']
            pred_word_list_strs, pred_word_lists_full_sentence, tgt_str = decode_entrance(tuned_probs[b],
                                                                                          tuned_mat_probs[b],
                                                                                          meta[b],
                                                                                          self._use_disco,
                                                                                          trigram_block,
                                                                                          sem_red_matrix,
                                                                                          pair_oracle,
                                                                                          stop_by_word_cnt,
                                                                                          min_pred_unit,
                                                                                          max_pred_unit,
                                                                                          self._step,
                                                                                          self._min_pred_unit,
                                                                                          self._max_pred_unit,
                                                                                          threshold_for_red_map
                                                                                          )

            if self._stop_by_word_count:
                for l in range(self.slot_num):
                    getattr(self, 'rouge_{}'.format(l))(pred="<q>".join(pred_word_list_strs[l]),
                                                        ref=tgt_str)
            else:
                if source_name == 'cnn':
                    pred_word_list_strs.insert(0, pred_word_list_strs[0])
                    pred_word_list_strs.pop(-1)
                    pred_word_lists_full_sentence.insert(0, pred_word_lists_full_sentence[0])
                    pred_word_lists_full_sentence.pop(-1)
                for l in range(min_pred_unit, max_pred_unit):
                    pred_word_list_strs[l] = [x for x in pred_word_list_strs[l] if len(x) > 1]
                    pred_word_lists_full_sentence[l] = [x for x in pred_word_lists_full_sentence[l] if len(x) > 1]
                    getattr(self, 'rouge_{}_{}'.format(l, threshold_for_red_map))(
                        pred="<q>".join(pred_word_list_strs[l]),
                        ref=tgt_str,
                        full_sent="<q>".join(pred_word_lists_full_sentence[l],

                                             ), idstr=doc_id
                    )
        return output_dict

    def ultra_fine_metrics(self, dict_of_rouge):
        best_key, best_val = "", -1
        # for ths in self._threshold_red_map:
        #     for l in  range(self._min_pred_unit, self._max_pred_unit):
        # threshold_slots = {}
        # pred_len_slots = {}
        # best_of_best = None
        # among all pred_unit, whose the best
        # among all threshold, whose the best
        # for key, val in dict_of_rouge.items():
        #     if not key.endswith("_1"):
        #         continue
        #     segs = key.split("_")
        #     pred_l = segs[1]
        #     thres = segs[2]
        #     if thres in threshold_slots:
        #         thre
        for key, val in dict_of_rouge.items():
            if key.endswith("_1"):
                if val > best_val:
                    best_val = val
                    best_key = key
        best_name = best_key.split("_")
        pred_len = int(best_name[1])
        thres = float(best_name[2])
        return "_".join(best_name[:3]), pred_len, thres

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # dict_of_rouge = self._rouge.get_metric(reset)
        dict_of_rouge = {}
        # print(reset)
        if reset:
            if self._stop_by_word_count:
                obj_list = [(getattr(self, 'rouge_{}'.format(l)),) for l in range(self.slot_num)]
            else:
                obj_list = [[(getattr(self, 'rouge_{}_{}'.format(l, ths)),) for l in
                             range(self._min_pred_unit, self._max_pred_unit)] for ths in self._threshold_red_map]
                obj_list = sum(obj_list, [])
            pool = multiprocessing.Pool(processes=10)
            results = pool.starmap(run_eval_worker, obj_list)
            pool.close()
            pool.join()


            if self._stop_by_word_count:

                for l in range(self.slot_num):
                    getattr(self, 'rouge_{}'.format(l)).reset()
            else:
                for l in range(self._min_pred_unit, self._max_pred_unit):
                    for ths in self._threshold_red_map:
                        getattr(self, 'rouge_{}_{}'.format(l, ths)).reset()

            for r in results:
                dict_of_rouge = {**dict_of_rouge, **r}
        else:
            if self._stop_by_word_count:
                for l in range(self.slot_num):
                    _d = getattr(self, 'rouge_{}'.format(l)).get_metric(reset)
                    dict_of_rouge = {**dict_of_rouge, **_d}
            else:
                for l in range(self._min_pred_unit, self._max_pred_unit):
                    for ths in self._threshold_red_map:
                        _d = getattr(self, 'rouge_{}_{}'.format(l, ths)).get_metric(reset)
                        dict_of_rouge = {**dict_of_rouge, **_d}
        best_name, pred_len, thres = self.ultra_fine_metrics(dict_of_rouge)

        if reset:
            print("--> Best_key: {}".format(best_name))
        metrics = {
            # 'accuracy': self._accuracy.get_metric(reset),
            'R_1': dict_of_rouge['{}_1'.format(best_name)],
            'R_2': dict_of_rouge['{}_2'.format(best_name)],
            'R_L': dict_of_rouge['{}_L'.format(best_name)],
            'L': pred_len,
            'T': thres
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

    # finetune = True
    finetune = False

    logger.info("AllenNLP version {}".format(allennlp.__version__))

    jsonnet_file = os.path.join(root, 'configs/baseline_bert.jsonnet')
    params = Params.from_file(jsonnet_file)
    print(params.params)

    serialization_dir = tempfile.mkdtemp(prefix=os.path.join(root, 'tmp_exps'))
    if finetune:
        # model_arch = '/datadrive/GETSum/cnndmfusion'
        # model_arch = '/datadrive/GETSum/cnndm_disco_cc'
        model_arch = '/datadrive/GETSum/tmp_expsyj8uupql'
        # model_arch = '/datadrive/GETSum/nyt_fusion_continue'

        fine_tune_model_from_file_paths(model_arch,
                                        os.path.join(root, 'configs/baseline_bert.jsonnet'),
                                        # os.path.join(root, 'configs/finetune.jsonnet'),
                                        serialization_dir
                                        )
    else:
        model = train_model(params, serialization_dir)
    print(serialization_dir)
