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

import string
from allennlp.common.util import START_SYMBOL, END_SYMBOL
import logging
import os, torch, pickle

from typing import Dict, Optional, List, Any

from overrides import overrides

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
                 lazy: bool,
                 bert_model_name: str,
                 token_indexers: Dict[str, TokenIndexer] = PretrainedBertIndexer("bert-base-uncased"),
                 debug: bool = False,
                 ) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._debug = debug
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # self.bert_tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())
        self.lowercase_input = "uncased" in bert_model_name
        logger.info("Finish Initializing of Dataset Reader")
        self.bert_lut = list(self._token_indexers['bert'].vocab.items())
        self.bert_lut = [x[0] for x in self.bert_lut]

    def _read(self, file_path):
        files = os.listdir(file_path)

        partition_name = identify_partition_name(file_path)
        if partition_name == 'test' and self._debug:
            logger.warning("debug mode only loads part of test set!")
            files = files[:2]
        random.seed(112312)
        random.shuffle(files)
        cnt = 0
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
                                            d['src_txt'],
                                            d['tgt_txt'],
                                            identify_partition_name(f)
                                            )
                # cnt += 1
                # if cnt > 2000:
                #     return

    @overrides
    def text_to_instance(self,
                         doc_text: List[int],  # Must have. List[int]
                         labels: List[int],  # label, binary supervision, position wise
                         segs: List[int],  # segments binary
                         clss: List[int],
                         src_txt: List[str],
                         tgt_txt: str,
                         type: str
                         ):

        # sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        # tokens1 = self.tokenizer.tokenize(sentence1)
        # print(len(doc_text))
        # print(doc_text)
        assert len(segs) > 0
        assert len(labels) > 0
        assert len(clss) > 0
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
        meta_field = MetadataField({
            "source": 'cnndm',
            "type": type,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt
        })
        fields = {"tokens": text_tokens,
                  "labels": labels,
                  "segs": segs,
                  "clss": clss,
                  "meta_field": meta_field
                  }
        return Instance(fields)

# if __name__ == '__main__':
#     dataset_reader = CNNDMDatasetReader()
#     x = dataset_reader._read()
#     for i in x:
#         print(i)
