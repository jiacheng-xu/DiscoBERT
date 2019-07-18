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

numpy.set_printoptions(threshold=sys.maxsize)
import string
from allennlp.common.util import START_SYMBOL, END_SYMBOL
import logging
import os, torch, pickle

from typing import Dict, Optional, List, Any

from overrides import overrides
import dgl

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
                 token_indexers: Dict[str, TokenIndexer] = PretrainedBertIndexer("bert-base-uncased"),
                 debug: bool = False,
                 ) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._token_indexers['bert'].max_pieces=1024
        self._debug = debug
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # self.bert_tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())
        self.lowercase_input = "uncased" in bert_model_name
        logger.info("Finish Initializing of Dataset Reader")
        self.bert_lut = list(self._token_indexers['bert'].vocab.items())
        self.bert_lut = [x[0] for x in self.bert_lut]

        self.train_pts = []
        random.seed(112312)

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
                                                d['tgt_txt'],
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
                    # print("yielding from {}".format(f))
                    # print(d)
                    yield self.text_to_instance(d['src'],
                                                d['labels'],
                                                d['segs'],
                                                d['clss'],
                                                d['sent_txt'],
                                                d['disco_txt'],
                                                d['tgt_txt'],
                                                d['d_labels'],
                                                d['d_span'],
                                                d['d_coref'],
                                                d['d_graph'],
                                                d['disco_dep'],
                                                d['doc_id'],
                                                identify_partition_name(f)
                                                )

        #
        # if partition_name == 'test' and self._debug:
        #     logger.warning("debug mode only loads part of test set!")
        #     files = files[:2]
        #
        # for f in files:
        #     dataset = torch.load(os.path.join(file_path, f))
        #     print('Loading dataset from %s, number of examples: %d' %
        #           (f, len(dataset)))
        #     logger.info('Loading dataset from %s, number of examples: %d' %
        #                 (f, len(dataset)))
        #     for d in dataset:
        #         # print("yielding from {}".format(f))
        #         # print(d)
        #         yield self.text_to_instance(d['src'],
        #                                     d['labels'],
        #                                     d['segs'],
        #                                     d['clss'],
        #                                     d['sent_txt'],
        #                                     d['disco_txt'],
        #                                     d['tgt_txt'],
        #                                     d['d_labels'],
        #                                     d['d_span'],
        #                                     d['d_coref'],
        #                                     identify_partition_name(f)
        #                                     )
        # cnt += 1
        # if cnt > 2000:
        #     return

    @overrides
    def text_to_instance(self,
                         doc_text: List[int],  # Must have. List[int]
                         labels: List[int],  # label, binary supervision, position wise
                         segs: List[int],  # segments binary
                         clss: List[int],
                         sent_txt: List[str],
                         disco_txt: List[str],
                         tgt_txt: str,
                         disco_label,
                         disco_span,
                         disco_coref,
                         disco_graph,
                         disco_dep,
                         doc_id: str,
                         spilit_type
                         ):

        # sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        # tokens1 = self.tokenizer.tokenize(sentence1)
        # print(len(doc_text))
        # print(doc_text)
        assert len(segs) > 0
        assert len(labels) > 0
        assert len(clss) > 0

        num_of_disco = len(disco_label[0])

        text_tokens = [Token(text=self.bert_lut[x], idx=x) for x in doc_text][1:-1]
        text_tokens = TextField(text_tokens, self._token_indexers
                                )
        #            2   3    4   3     5     6   8      9    2   14   12
        # sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        # tokens1 = self.bert_tokenizer.tokenize(sentence1)
        # text_tokens = tokens1
        # text_tokens = TextField(text_tokens, {"bert": self._token_indexers})

        labels = ArrayField(np.asarray(labels), padding_value=-1, dtype=np.int)
        segs = ArrayField(np.asarray(segs), padding_value=0, dtype=np.int)  # TODO -1 or 0?
        clss = ArrayField(np.asarray(clss), padding_value=-1, dtype=np.int)

        disco_label = ArrayField(np.asarray(disco_label), padding_value=-1, dtype=np.int)
        disco_span = ArrayField(np.asarray(disco_span), padding_value=-1, dtype=np.int)

        disco_coref = [x for x in disco_coref if x[0] != x[1]]
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
        empty = np.zeros((num_of_disco, num_of_disco), dtype=float)
        #########
        for cor in disco_coref:
            x, y = cor
            empty[x][y] += 1
            empty[y][x] += 1
        eye = np.eye(num_of_disco, dtype=float)
        np_graph = eye + empty
        coref_graph = ArrayField(np_graph, padding_value=0, dtype=np.float32)
        #########

        ########
        # disco graph
        dis_graph = np.zeros((num_of_disco, num_of_disco), dtype=float)
        for rst in disco_graph:
            rst_src, rst_tgt = rst[0], rst[1]
            if rst_src < num_of_disco and rst_tgt < num_of_disco:
                dis_graph[rst_src][rst_tgt] = 1
        # if random.random() < 0.01:
        #     print(dis_graph)
        dis_graph = ArrayField(dis_graph, padding_value=0, dtype=np.float32)
        ########

        meta_field = MetadataField({
            "source": 'cnndm',
            "type": spilit_type,
            "sent_txt": sent_txt,
            "disco_txt": disco_txt,
            "tgt_txt": tgt_txt,
            'disco_dep': disco_dep,
            'doc_id':doc_id

            # "coref_graph": coref_graph
        })
        fields = {"tokens": text_tokens,
                  "labels": labels,
                  "segs": segs,
                  "clss": clss,
                  "meta_field": meta_field,
                  "disco_label": disco_label,
                  "disco_span": disco_span,
                  'disco_coref_graph': coref_graph,
                  'disco_rst_graph': dis_graph
                  }
        return Instance(fields)


if __name__ == '__main__':
    dataset_reader = CNNDMDatasetReader()
    x = dataset_reader._read("/datadrive/data/cnn/chunk")
    for i in x:
        print(i)
