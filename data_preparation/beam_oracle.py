import itertools
from typing import List
import numpy

# from thesaurus import Word
# import thesaurus as th
import math
import allennlp
import math

# from allennlp.data.tokenizers.word_stemmer import PorterStemmer

flatten = lambda l: [item for sublist in l for item in sublist]
from nltk import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

ps = PorterStemmer()
cache_for_th = {}


def remove_duplicate_tok(inp: List[str]):
    # replace duplicates with SPTOK = "_DUPLICATE_"
    for index in range(len(inp)):
        word = inp[
            index
        ]
        cnts = inp.count(word)
        if cnts > 1:
            indices = [i for i, x in enumerate(inp) if x == word][1:]
            inp = ["_DUPLICATE_" if i in indices else x for i, x in enumerate(inp)]
    # print(inp)
    return inp


def replace_w_morphy(inp: List[str]):
    output = []
    for x in inp:
        out = wn.morphy(x)
        if out:
            output.append(out)
        else:
            output.append(x)
    return output


class DocumentOracleDerivation(object):
    def __init__(self,
                 min_combination_num: int = 3,
                 max_combination_num: int = 5,
                 rm_stop_word: bool = True,
                 synonyms: bool = True,
                 stem: bool = False,
                 tokenization: bool = True,
                 beam_sz: int = 5,
                 candidate_percent: float = 1.0):
        self.min_combination_num = min_combination_num
        self.max_combination_num = max_combination_num
        self.rm_stop_word = rm_stop_word
        self.stem = stem
        self.tokenization = tokenization
        self.beam_sz = beam_sz
        self.candidate_percent = candidate_percent
        if self.stem:
            self.stemmer = PorterStemmer().stem_word
        else:
            self.stemmer = lambda x: x
        self.synonyms = synonyms
        if self.tokenization:
            from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
            self.tokenizer = WordTokenizer()
        if self.rm_stop_word:
            self.stop_words = list(set(stopwords.words('english'))) + [x for x in string.punctuation] + ['``', '\'\'']
        else:
            self.stop_words = []

    def get_rouge_w_annotation_ready_to_use(self, gold_tokens: List[str],
                                            pred_tokens: List[str]):
        gold_lower = list(set([x.lower() for x in gold_tokens]))
        gold_wo_stop = [x for x in gold_lower if x not in self.stop_words]  # change of index
        gold_wo_stop = replace_w_morphy(gold_wo_stop)
        gold_stem = [ps.stem(x) for x in gold_wo_stop]

        pred_lower = list([x.lower() for x in pred_tokens])

        pred_lower = replace_w_morphy(pred_lower)
        pred_lower = remove_duplicate_tok(pred_lower)

        pred_stem = [ps.stem(x) for x in pred_lower]
        pred_stem = remove_duplicate_tok(pred_stem)
        size_of_gold = len(gold_stem)
        size_of_pred = len(pred_stem)

        gold_key, gold_value = [], []
        for idx, word in enumerate(gold_wo_stop):
            # for one gold word, we have a minigroup
            _tmp = []
            if word in pred_lower:
                _tmp.append(word)
            elif word in pred_stem:
                _tmp.append(word)
            elif gold_stem[idx] in pred_lower:
                _tmp.append(gold_stem[idx])
            elif gold_stem[idx] in pred_stem:
                _tmp.append(gold_stem[idx])

            # if word or stm word could match, we don't need to search syn
            if _tmp != []:
                _tmp = _tmp[0]
                gold_key.append(_tmp)
                gold_value.append(1)
            else:
                if word not in cache_for_th:
                    try:
                        cache_for_th[word] = flatten(th.Word(word).synonyms('all', relevance=[3]))
                    except:
                        cache_for_th[word] = []

                if gold_stem[idx] not in cache_for_th:
                    try:
                        cache_for_th[gold_stem[idx]] = flatten(
                            th.Word(gold_stem[idx]).synonyms('all', relevance=[3]))
                    except:
                        cache_for_th[gold_stem[idx]] = []
                syn = cache_for_th[word]
                syn_stem = cache_for_th[gold_stem[idx]]
                syn = list(set(syn + syn_stem))
                # print(syn)
                l_syn = len(syn)
                if l_syn != 0:
                    gold_key += syn
                    gold_value += [float(1 / l_syn)] * l_syn

        gold_tokens = [ps.stem(x) for x in gold_key]
        # pred_set = set(pred)
        # comp intersection
        vs = 0
        key_index = []
        for p_idx in range(len(pred_lower)):
            p_word = pred_lower[p_idx]
            p_stem_word = pred_stem[p_idx]

            if p_word in gold_key:
                idx = gold_key.index(p_word)
                v = gold_value[idx]
                vs += v
                key_index.append(p_idx)
            elif p_stem_word in gold_tokens:
                idx = gold_tokens.index(p_stem_word)
                v = gold_value[idx]
                vs += v
                key_index.append(p_idx)

        rouge_recall_1 = 0
        if size_of_gold != 0:
            rouge_recall_1 = vs / float(size_of_gold)
        rouge_pre_1 = 0
        if size_of_pred != 0:
            rouge_pre_1 = vs / float(size_of_pred)
        # print(rouge_recall_1, rouge_pre_1)
        # assert rouge_recall_1 <= 1
        # assert rouge_pre_1 <= 1
        if random.random() < 0.00001:
            print("Recall: {}\tPre: {}".format(rouge_recall_1, rouge_pre_1))
            print(pred_tokens)
        customed_recall = rouge_recall_1 + rouge_pre_1 * 0.01 - 0.01
        f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
                rouge_recall_1 + rouge_pre_1)
        return customed_recall, f1, key_index
        # f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
        #         rouge_recall_1 + rouge_pre_1)
        # f1 = rouge_recall_1 * 5 + rouge_pre_1

    def comp_num_seg_out_of_p_sent_beam(self, _filtered_doc_list,
                                        num_sent_in_combination,
                                        target_ref_sum_list,
                                        map_from_new_to_ori_idx) -> dict:
        beam: List[dict] = []
        if len(_filtered_doc_list) < num_sent_in_combination:
            return {"nlabel": num_sent_in_combination,
                    "data": {},
                    "best": None
                    }

        combs = list(range(0, len(_filtered_doc_list)))
        # _num_edu seq_len
        cur_beam = {
            "in": [],
            "todo": combs,
            "val": 0
        }
        beam.append(cur_beam)
        for t in range(num_sent_in_combination):
            dict_pattern = {}
            # compute top beam_sz for every beam
            global_board = []
            for b in beam:
                already_in_beam = b['in']
                todo = b['todo']

                leaderboard = {}
                for to_add in todo:
                    after_add = already_in_beam + [to_add]
                    candidate_doc_list = list(itertools.chain.from_iterable([_filtered_doc_list[i] for i in after_add]))
                    # average_f_score = self.get_approximate_rouge(target_ref_sum_list, candidate_doc_list)
                    _, average_f_score, _ = self.get_rouge_w_annotation_ready_to_use(gold_tokens=target_ref_sum_list,
                                                                                     pred_tokens=candidate_doc_list)
                    leaderboard[to_add] = average_f_score
                sorted_beam = [(k, leaderboard[k]) for k in sorted(leaderboard, key=leaderboard.get, reverse=True)]

                for it in sorted_beam:
                    new_in = already_in_beam + [it[0]]
                    new_in.sort()
                    str_new_in = [str(x) for x in new_in]
                    if '_'.join(str_new_in) in dict_pattern:
                        continue
                    else:
                        dict_pattern['_'.join(str_new_in)] = True
                    new_list = todo.copy()
                    new_list.remove(it[0])
                    _beam = {
                        "in": new_in,
                        "todo": new_list,
                        "val": it[1]
                    }
                    global_board.append(_beam)
            # merge and get the top beam_sz among all

            sorted_global_board = sorted(global_board, key=lambda x: x["val"], reverse=True)

            _cnt = 0
            check_dict = []
            beam_waitlist = []
            for it in sorted_global_board:
                str_in = sorted(it['in'])
                str_in = [str(x) for x in str_in]
                _tmp_key = '_'.join(str_in)
                if _tmp_key in check_dict:
                    continue
                else:
                    beam_waitlist.append(it)
                    check_dict.append(_tmp_key)
                _cnt += 1
                if _cnt >= self.beam_sz:
                    break
            beam = beam_waitlist
        # if len(beam) < 2:
        #     print(len(_filtered_doc_list))
        #     print(_num_edu)
        # Write oracle to a string like: 0.4 0.3 0.4
        _comb_bag = {}
        for it in beam:
            n_comb = it['in']
            n_comb.sort()
            n_comb_original = [map_from_new_to_ori_idx[a] for a in n_comb]
            n_comb_original.sort()  # json label
            n_comb_original = [int(x) for x in n_comb_original]
            candidate_doc_list = list(itertools.chain.from_iterable([_filtered_doc_list[i] for i in n_comb]))
            # f1 = self.get_approximate_rouge(target_ref_sum_list, candidate_doc_list)
            _, f1, _ = self.get_rouge_w_annotation_ready_to_use(target_ref_sum_list, candidate_doc_list)

            # f_avg = (f1 + f2 + fl) / 3
            _comb_bag[f1] = {"label": n_comb_original,
                             "R1": f1,
                             "nlabel": num_sent_in_combination}
        # print(len(_comb_bag))
        if len(_comb_bag) == 0:
            return {"nlabel": num_sent_in_combination,
                    "data": {},
                    "best": None
                    }
        else:
            best_key = sorted(_comb_bag.keys(), reverse=True)[0]
            rt_dict = {"nlabel": num_sent_in_combination,
                       "data": _comb_bag,
                       "best": _comb_bag[best_key]
                       }
            return rt_dict

    def derive_doc_oracle(self, doc_list: List[str],
                          ref_sum: str,
                          prefix_summary: str = ""
                          ):
        processed_doc_list, processed_ref_sum_str, processed_prefix_sum_str = [], '', ''
        if self.tokenization:
            token_doc_list = self.tokenizer.batch_tokenize(doc_list)
            for doc in token_doc_list:
                processed_doc_list.append([word.text for word in doc])
            processed_ref_sum_list = [w.text for w in self.tokenizer.tokenize(ref_sum)]
            processed_prefix_sum_list = [w.text for w in self.tokenizer.tokenize(prefix_summary)]
        else:
            processed_doc_list = [d.split(" ") for d in doc_list]
            processed_ref_sum_list = ref_sum.split(" ")
            processed_prefix_sum_list = prefix_summary.split(" ")
        processed_doc_list = [[x.lower() for x in sent] for sent in processed_doc_list]
        processed_ref_sum_list = [x.lower() for x in processed_ref_sum_list]
        processed_prefix_sum_list = [x.lower() for x in processed_prefix_sum_list]
        if self.rm_stop_word:
            processed_doc_list = [[x for x in sent if x not in self.stop_words] for sent in processed_doc_list]
            processed_ref_sum_list = [x for x in processed_ref_sum_list if x not in self.stop_words]
            processed_prefix_sum_list = [x for x in processed_prefix_sum_list if x not in self.stop_words]

        target_ref_sum_list = [x for x in processed_ref_sum_list if x not in processed_prefix_sum_list]

        # preprocessing finished
        filtered_doc_list, map_from_new_to_ori_idx = self.pre_prune(processed_doc_list, target_ref_sum_list)
        combination_data_dict = {}
        for num_sent_in_combination in range(self.min_combination_num, self.max_combination_num):
            combination_data = self.comp_num_seg_out_of_p_sent_beam(_filtered_doc_list=filtered_doc_list,
                                                                    num_sent_in_combination=num_sent_in_combination,
                                                                    target_ref_sum_list=target_ref_sum_list,
                                                                    map_from_new_to_ori_idx=map_from_new_to_ori_idx)

            combination_data_dict[num_sent_in_combination] = combination_data
        return combination_data_dict

    def pre_prune(self, list_of_doc: List[List[str]],
                  ref_sum: List[str]
                  ):
        keep_candidate_num = math.ceil(len(list_of_doc) * self.candidate_percent)
        # f_score_list = [self.get_approximate_rouge(ref_sum, x) for x in list_of_doc]
        f_score_list = [self.get_rouge_w_annotation_ready_to_use(ref_sum, x)[1] for x in list_of_doc]
        top_p_sent_idx = numpy.argsort(f_score_list)[-keep_candidate_num:]

        map_from_new_to_ori_idx = []
        # filter
        filtered_doc_list = []
        for i in range(len(top_p_sent_idx)):
            filtered_doc_list.append(list_of_doc[top_p_sent_idx[i]])
            map_from_new_to_ori_idx.append(top_p_sent_idx[i])
        return filtered_doc_list, map_from_new_to_ori_idx


import json
import string


def fix_perioids(inp: List[str]) -> List[str]:
    punc_str = string.punctuation
    out = []
    for i in inp:
        i = i.strip()
        if len(i) > 3:
            if punc_str.find(i[-1]) > -1:
                out.append(i)
            else:
                out.append(i + ".")
    return out


import random

if __name__ == '__main__':
    single_orac = DocumentOracleDerivation(min_combination_num=1, max_combination_num=2)
    ora = single_orac.derive_doc_oracle([
        "Traffic is still far from gridlock, but electric bikes have now joined the taxis, a growing fleet of private cars, and the Soviet-era trolleybuses that have plied the capital for decades. ",
        " Traffic is getting busier in Pyongyang, which last year began laying out its first dedicated bicycle lanes. "
    ],
        "Trading companies")
    print(ora)
