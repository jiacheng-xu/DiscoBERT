import json
import os

import gc
import torch
from pytorch_pretrained_bert import BertTokenizer

from data_preparation.data_structure import MSBertData
from data_preparation.nlpyang_others_logging import logger
from data_preparation.nlpyang_others_utils import clean

glob_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

import itertools


class SentUnit():
    def __init__(self, sent_index, raw_words, list_of_bpes, discourse_bag):
        self.sent_index = sent_index
        self.raw_words = raw_words
        self.bpes = list(itertools.chain(*list_of_bpes))
        self.prefix_len = -1
        self.discourse_bag = discourse_bag
        # mentions  corefs

    def get_bpe_w_cls_sep(self):
        return ['[CLS]'] + self.bpes + ['[SEP]']

    def get_length_w_pad(self):
        return len(self.bpes) + 2


class DiscourseUnit():
    def __init__(self, unq_idx, sent_idx, rel_start, rel_end):
        self.unq_idx = unq_idx
        self.sent_idx = sent_idx
        self.original_start_in_sent = rel_start
        self.original_end_in_sent = rel_end
        self.raw_words = []
        self.bert_word_pieces = []
        self.mentions = []
        self.corefs = []

    def add_dep_info(self, tree_node):
        self.disco_id = tree_node[0] - 1
        # print(self.disco_id)
        # print(self.unq_idx)
        assert self.disco_id == self.unq_idx
        self.disco_type = tree_node[1]
        self.disco_rel = tree_node[2]
        self.disco_dep = tree_node[3] - 1

    def get_readable_words_as_list(self):
        return self.raw_words

    def get_readable_words_as_str(self):
        return " ".join(self.raw_words)

    # def get_bpe_w_cls_sep(self):
    #     return ['[CLS]'] + self.bert_word_pieces + ['SEP']

    def get_bpe_only(self):
        return self.bert_word_pieces

    def add_word(self, word_to_add, tokenizer=glob_bert_tokenizer):
        self.raw_words.append(word_to_add)
        self.bert_word_pieces += tokenizer.tokenize(word_to_add)

    def add_mention(self, word_index):
        self.mentions.append("{}_{}".format(self.sent_idx, word_index))

    def add_coref(self, coref_list):
        s = coref_list[0]
        w = coref_list[1]
        _t = "{}_{}".format(s, w)
        if _t not in self.mentions:
            self.corefs.append(_t)

    def get_original_length(self):
        return len(self.raw_words)

    def get_bert_wp_length(self):
        return len(self.bert_word_pieces)

    def get_original_location_sent(self):
        return self.sent_idx

    def respond_broadcast(self, sent_idx, word_idx) -> bool:
        if self.sent_idx == sent_idx \
                and word_idx <= self.original_end_in_sent \
                and word_idx >= self.original_start_in_sent:
            return True
        else:
            return False


def get_next_node(inp):
    for x in inp:
        yield x


def MS_formate_to_bert(params):
    length_limit = 510
    read_json_file, wt_pt_file, oracle_mode, oracle_sent_num, min_src_ntokens, max_src_ntokens, min_nsents, max_nsents = params
    # print(read_json_file)
    # TODO keep file exist check
    if os.path.exists(wt_pt_file):
        logger.info('Ignore %s' % wt_pt_file)
        return
    print("working on {}".format(wt_pt_file))
    bert_data = MSBertData(min_src_ntokens, max_src_ntokens, min_nsents, max_nsents)

    logger.info('Processing %s' % read_json_file)
    jobs = json.load(open(read_json_file))
    datasets = []
    for d in jobs:
        disco_node = d['disco_node']
        gen = get_next_node(disco_node)
        if len(disco_node) == 0:
            print(d)
            continue
        # print(len(disco_node))
        disco_graph_links = d['disco_graph_links']

        span, tgt = d['disco_span'], d['tgt']
        sent, doc_id, coref = d['sent'], d['doc_id'], d['coref']

        # First of all, assemble data and  LENGTH truncation
        budget = 0
        disco_bag = []
        sent_bag = []
        original_disco_txt_list_of_str = []
        for idx in range(len(sent)):

            this_sent = sent[idx]
            this_disco = span[idx]
            this_tokens = this_sent['tokens']
            this_tokens = [clean(x.lower()) for x in this_tokens]
            this_coref = this_sent['corefs']
            original_word_len = len(this_tokens)

            tmp_disco_bag = []
            for disc in this_disco:

                tree_node = next(gen)
                start, end = disc
                disc_piece = DiscourseUnit(len(disco_bag) + len(tmp_disco_bag), idx, rel_start=start, rel_end=end)
                disc_piece.add_dep_info(tree_node)
                for jdx in range(start, end + 1):
                    _toks = this_tokens[jdx]

                    disc_piece.add_word(_toks)

                    # look at word jdx, see if any coref mentions applied.
                    _cor = this_coref[jdx]
                    if _cor != []:
                        disc_piece.add_mention(jdx)  # add the orinigla index of the word in the sentence
                        for _c in _cor:
                            disc_piece.add_coref(_c)
                    # finish loading coref
                tmp_disco_bag.append(disc_piece)
                budget += disc_piece.get_bert_wp_length()
            budget += 2
            if budget > length_limit:
                break
            else:
                disco_bag += tmp_disco_bag
                original_disco_txt_list_of_str += [x.get_readable_words_as_list() for x in tmp_disco_bag]
                s = SentUnit(idx, this_tokens, [x.bert_word_pieces for x in tmp_disco_bag], tmp_disco_bag)
                sent_bag.append(s)

        effective_disco_number = len(disco_bag)
        # clean disco_graph_links
        disco_graph_links = [(tup[0] - 1, tup[1] - 1, tup[2]) for tup in disco_graph_links if
                             (tup[0] <= effective_disco_number and tup[1] <= effective_disco_number)]
        disc_oracle_ids, disc_spans, disc_coref = bert_data.preprocess_disc(disco_bag, tgt, 4)
        src_tok_index, sent_oracle_labels, segments_ids, \
        cls_ids, original_sent_txt_list_of_str, tgt_txt = bert_data.preprocess_sent(sent_bag, summary=tgt)
        # TO have: src_subtoken_idxs [for bert encoder], labels[sent level and discourse level],
        # segments_ids[for bert encoder],
        # cls_ids[for sent level],
        # span indexs [ for discourse level]
        # entity coref linking edge [ sent level and discourse level]
        # discourse linking edge [discourse level only]
        # src_txt, tgt_txt

        # provide two versions, one based on discourse, one without.
        # w. multiple oracle

        # prepare discourse data
        # oracle is computed based on discourse

        # prepare sent data

        b_data_dict = {"src": src_tok_index,
                       "labels": sent_oracle_labels,
                       "segs": segments_ids,
                       'clss': cls_ids,
                       'sent_txt': original_sent_txt_list_of_str,
                       'disco_txt': original_disco_txt_list_of_str,
                       "tgt_txt": tgt_txt,
                       'd_labels': disc_oracle_ids,
                       'd_span': disc_spans,
                       'd_coref': disc_coref,
                       'd_graph': disco_graph_links
                       }
        if len(src_tok_index) < 15 or len(tgt_txt) < 3:
            continue
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % wt_pt_file)
    torch.save(datasets, wt_pt_file)
    datasets = []
    gc.collect()
