import nltk
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import *
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter

from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder
import sys
import numpy
from nltk.tokenize import TweetTokenizer

from model.sem_red_map import MapKiosk

tknzr = TweetTokenizer()
from data_preparation.search_algo import original_greedy_selection

numpy.set_printoptions(threshold=sys.maxsize)
import string
from allennlp.common.util import START_SYMBOL, END_SYMBOL
import logging
import os, torch, pickle

from typing import Dict, Optional, List, Any

from overrides import overrides
import dgl


def label_filter(labs):
    rt_list = []
    cur_min_cnt = 100
    for l in labs:
        s = sum(l)
        if s < cur_min_cnt:
            cur_min_cnt = s
            rt_list.insert(0, l)
        else:
            rt_list.append(l)
    return rt_list
    pass


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import numpy as np
import random, glob, gc
from allennlp.data.tokenizers.token import Token


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer


def identify_partition_name(full_name):
    if 'train' in full_name:
        return 'train'
    elif 'valid' in full_name or 'dev' in full_name:
        return 'valid'
    elif 'test' in full_name:
        return 'test'


from pytorch_pretrained_bert.tokenization import BertTokenizer


@DatasetReader.register("cnndm")
class CNNDMDatasetReader(DatasetReader):
    def __init__(self,

                 lazy: bool = True,
                 bert_model_name: str = 'bert-base-uncased',
                 max_bpe: int = None,
                 token_indexers: Dict[str, TokenIndexer] = PretrainedBertIndexer("bert-base-uncased"),
                 debug: bool = False,
                 bertsum_oracle: bool = True,
                 semantic_red_map: bool = True,
                 semantic_red_map_key: List[str] = None
                 ) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if max_bpe is not None:
            self._token_indexers['bert'].max_pieces = max_bpe
        self._debug = debug
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # self.bert_tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())
        self.lowercase_input = "uncased" in bert_model_name
        logger.info("Finish Initializing of Dataset Reader")
        self.bert_lut = list(self._token_indexers['bert'].vocab.items())
        self.bert_lut = [x[0] for x in self.bert_lut]
        self.max_bpe = max_bpe
        self.train_pts = []
        self.bertsum_oracle = bertsum_oracle

        self.semantic_red_map = semantic_red_map
        if semantic_red_map:
            self.map_kiosk = MapKiosk(semantic_red_map_key)

        random.seed(1112)

    def boil_pivot_table(self, sent_txt):
        pass

    def refill_data(self, fpath):
        partition_name = identify_partition_name(fpath)
        if hasattr(self, partition_name):
            if getattr(self, partition_name) == []:
                print("start a new round of loading training data")
                files = os.listdir(fpath)
                files = [f for f in files if f.endswith("pt")]
                random.shuffle(files)
                setattr(self, partition_name, files)
        else:
            print("start a new round of loading training data")
            files = os.listdir(fpath)
            files = [f for f in files if f.endswith("pt")]
            random.shuffle(files)
            setattr(self, partition_name, files)

    def yield_data(self, part_name, fpath):
        while True:
            self.refill_data(fpath)
            yield getattr(self, part_name).pop()

    def _read(self, file_path):

        # see and refill data files

        partition_name = identify_partition_name(file_path)

        if partition_name == 'train':
            for f in self.yield_data(partition_name, file_path):
                dataset = torch.load(os.path.join(file_path, f))
                print('Loading dataset from %s, number of examples: %d' %
                      (f, len(dataset)))
                logger.info('Loading dataset from %s, number of examples: %d' %
                            (f, len(dataset)))
                for d in dataset:
                    yield self.text_to_instance(d['src'],
                                                d['labels'],
                                                d['segs'],
                                                d['clss'],
                                                d['sent_txt'],
                                                d['disco_txt'],
                                                d['tgt_list_str'],
                                                d['tgt_tok_list_list_str'],
                                                d['d_labels'],
                                                d['d_span'],
                                                d['d_coref'],
                                                d['d_graph'],
                                                d['disco_dep'],
                                                d['doc_id'],
                                                identify_partition_name(f)
                                                )
        else:
            files = os.listdir(file_path)
            files = [f for f in files if f.endswith("pt")]
            if self._debug:
                logger.warning("debug mode only loads part of test set!")
                files = files[:1]
            for f in files:
                dataset = torch.load(os.path.join(file_path, f))
                print('Loading dataset from %s, number of examples: %d' %
                      (f, len(dataset)))
                logger.info('Loading dataset from %s, number of examples: %d' %
                            (f, len(dataset)))
                for d in dataset:
                    yield self.text_to_instance(d['src'],
                                                d['labels'],
                                                d['segs'],
                                                d['clss'],
                                                d['sent_txt'],
                                                d['disco_txt'],
                                                d['tgt_list_str'],
                                                d['tgt_tok_list_list_str'],
                                                d['d_labels'],
                                                d['d_span'],
                                                d['d_coref'],
                                                d['d_graph'],
                                                d['disco_dep'],
                                                d['doc_id'],
                                                identify_partition_name(f)
                                                )

    def create_disco_coref(self, disco_coref, num_of_disco):
        disco_coref = [x for x in disco_coref if x[0] != x[1]]
        coref_graph_as_list_of_tuple = [(x, x) for x in range(num_of_disco)]

        # DGL
        ##########
        # G = dgl.DGLGraph()
        # G.add_nodes(num_of_disco)
        #
        # full_disco_coref = disco_coref + [(x[1], x[0]) for x in disco_coref]
        # src, dst = tuple(zip(*full_disco_coref))
        # G.add_edges(src, dst)
        # G.add_edges(G.nodes(), G.nodes())
        # print(G.edata)
        ############

        # handmade
        # empty = np.zeros((num_of_disco, num_of_disco), dtype=float)
        #########
        for cor in disco_coref:
            x, y = cor
            if x < num_of_disco and y < num_of_disco:
                coref_graph_as_list_of_tuple.append((x, y))
                coref_graph_as_list_of_tuple.append((y, x))
                # empty[x][y] += 1
                # empty[y][x] += 1
        # eye = np.eye(num_of_disco, dtype=float)
        # np_graph = eye + empty
        # coref_graph = ArrayField(np_graph, padding_value=0, dtype=np.float32)
        #########

        return coref_graph_as_list_of_tuple

    def create_disco_graph(self, disco_graph, num_of_disco: int) -> List[tuple]:

        ########
        # disco graph
        dis_graph_as_list_of_tuple = []
        # dis_graph = np.zeros((num_of_disco, num_of_disco), dtype=float)
        for rst in disco_graph:
            rst_src, rst_tgt = rst[0], rst[1]
            if rst_src < num_of_disco and rst_tgt < num_of_disco:
                # dis_graph[rst_src][rst_tgt] = 1
                dis_graph_as_list_of_tuple.append((rst_src, rst_tgt))
        # dis_graph = ArrayField(dis_graph, padding_value=0, dtype=np.float32)

        return dis_graph_as_list_of_tuple

    def map_disco_to_sent(self, disco_span: List[tuple]):
        map_to_sent = [0 for _ in range(len(disco_span))]
        curret_sent = 0
        current_idx = 1
        for idx, disco in enumerate(disco_span):
            if disco[0] == current_idx:
                map_to_sent[idx] = curret_sent
            else:
                curret_sent += 1
                map_to_sent[idx] = curret_sent
            current_idx = disco[1]
        return map_to_sent

    @overrides
    def text_to_instance(self,
                         doc_text: List[int],  # Must have. List[int]
                         labels: List[int],  # label, binary supervision, position wise
                         segs: List[int],  # segments binary
                         clss: List[int],
                         sent_txt: List[str],
                         disco_txt: List[str],
                         tgt_list_str: List[str],
                         tgt_tok_list_list_str: List[List[str]],
                         disco_label,
                         disco_span,
                         disco_coref,
                         disco_graph,
                         disco_dep,
                         doc_id: str,
                         spilit_type
                         ):
        assert len(segs) > 0
        assert len(labels) > 0
        assert len(clss) > 0
        if self.max_bpe < 768:
            clss = [x for x in clss if x < self.max_bpe]
            doc_text = doc_text[:self.max_bpe]
            segs = segs[:self.max_bpe]
            actual_sent_len = len(clss)
            labels = [l[:actual_sent_len] for l in labels]

            # disco part
            disco_span = [x for x in disco_span if x[1] < self.max_bpe]
            num_of_disco = len(disco_span)
            disco_label = [l[:num_of_disco] for l in disco_label]
        else:
            actual_sent_len = len(sent_txt)

        num_of_disco = len(disco_label[0])

        text_tokens = [Token(text=self.bert_lut[x], idx=x) for x in doc_text][1:-1]
        text_tokens = TextField(text_tokens, self._token_indexers
                                )

        # sentence
        if self.semantic_red_map:
            maps = self.map_kiosk.single_entry_entrance(sent_txt[:actual_sent_len], tgt_tok_list_list_str)
            for k, v in maps.items():
                _v = ArrayField(np.asarray(v), padding_value=-1, dtype=np.int)
                maps[k] = _v
        else:
            maps = {}
        #            2   3    4   3     5     6   8      9    2   14   12
        # sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        # tokens1 = self.bert_tokenizer.tokenize(sentence1)
        if self.bertsum_oracle:
            labels = original_greedy_selection(sent_txt[:actual_sent_len], tgt_tok_list_list_str, 3)
            z = np.zeros((1, actual_sent_len))
            for l in labels:
                z[0][l] = 1
            labels = z
        # else:
        #     labels = label_filter(labels)

        labels = ArrayField(np.asarray(labels), padding_value=-1, dtype=np.int)
        segs = ArrayField(np.asarray(segs), padding_value=0, dtype=np.int)  # TODO -1 or 0?
        clss = ArrayField(np.asarray(clss), padding_value=-1, dtype=np.int)

        disco_label = label_filter(disco_label)
        disco_label = ArrayField(np.asarray(disco_label), padding_value=-1, dtype=np.int)

        disco_map_to_sent: List[int] = self.map_disco_to_sent(disco_span)
        disco_span = ArrayField(np.asarray(disco_span), padding_value=-1, dtype=np.int)

        coref_graph = self.create_disco_coref(
            disco_coref, num_of_disco
        )
        dis_graph = self.create_disco_graph(
            disco_graph, num_of_disco
        )
        meta_field = MetadataField({
            "source": 'cnndm',
            "type": spilit_type,
            "sent_txt": sent_txt,
            "disco_txt": disco_txt,
            "tgt_txt": "<q>".join(tgt_list_str),
            'disco_dep': disco_dep,
            'doc_id': doc_id,
            'disco_rst_graph': dis_graph,
            'disco_coref_graph': coref_graph,
            'disco_map_to_sent': disco_map_to_sent

            # "coref_graph": coref_graph
        })
        fields = {"tokens": text_tokens,
                  "labels": labels,
                  "segs": segs,
                  "clss": clss,
                  "meta_field": meta_field,
                  "disco_label": disco_label,
                  "disco_span": disco_span,
                  # 'disco_coref_graph': coref_graph,
                  # 'disco_rst_graph': dis_graph
                  }

        fields = {**maps, **fields}
        return Instance(fields)


if __name__ == '__main__':
    dataset_reader = CNNDMDatasetReader()
    x = dataset_reader._read("/datadrive/data/cnn/chunk")
    for i in x:
        print(i)
