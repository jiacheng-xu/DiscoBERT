import collections
import itertools
import string
from nltk import PorterStemmer
import nltk
from data_preparation.nlpyang_utils import _get_word_ngrams, _get_ngrams
from nltk.corpus import wordnet as wn
import re
from itertools import islice

from data_preparation.search_algo import cal_rouge

try:
    from nltk.corpus import stopwords

    stopwords.words('english')
except:
    nltk.download('stopwords')
import math
import numpy

ps = PorterStemmer()

from typing import List

flatten = lambda l: [item for sublist in l for item in sublist]
import multiprocessing

class DocumentOracleDerivation(object):
    def __init__(self,
                 mixed_combination: bool,
                 min_combination_num: int = 1,
                 max_combination_num: int = 8,
                 rm_stop_word: bool = True,
                 stem: bool = False,
                 morphy: bool = False,
                 tokenization: bool = True,
                 beam_sz: int = 5,
                 prune_candidate_percent: float = 0.4
                 ):
        self.mixed_combination = mixed_combination
        self.min_combination_num = min_combination_num
        self.max_combination_num = max_combination_num
        self.rm_stop_word = rm_stop_word
        self.stem = stem
        self.tokenization = tokenization
        self.beam_sz = beam_sz
        self.prune_candidate_percent = prune_candidate_percent
        if self.stem:
            self.stemmer = PorterStemmer().stem_word
        else:
            self.stemmer = lambda x: x

        self.morphy = morphy

        if self.tokenization:
            from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
            self.tokenizer = WordTokenizer()
        if self.rm_stop_word:
            self.stop_words = list(set(stopwords.words('english'))) + [x for x in string.punctuation] + ['``', '\'\'']
        else:
            self.stop_words = []

    def derive_doc_oracle(self,
                          doc_list: List[str],
                          ref_sum: str,
                          prefix_summary: str = "",
                          ):
        # return a dict where key=rouge-f1 and value= [0,0,0,1,0,1,0,...] same size as doc_list
        # processed_doc_list, processed_ref_sum_str, processed_prefix_sum_str = [], '', ''
        len_of_doc = len(doc_list)
        processed_doc_list = [self._rouge_clean(x) for x in doc_list]
        processed_ref_sum_str = self._rouge_clean(ref_sum)
        processed_prefix_sum_str = self._rouge_clean(prefix_summary)
        if self.tokenization:
            new_processed_doc_list = []
            token_doc_list = self.tokenizer.batch_tokenize(processed_doc_list)
            for doc in token_doc_list:
                new_processed_doc_list.append([word.text for word in doc])
            processed_doc_list = new_processed_doc_list
            processed_ref_sum_list = [w.text for w in self.tokenizer.tokenize(processed_ref_sum_str)]
            processed_prefix_sum_list = [w.text for w in self.tokenizer.tokenize(processed_prefix_sum_str)]
        else:
            processed_doc_list = [d.split(" ") for d in processed_doc_list]
            processed_ref_sum_list = processed_ref_sum_str.split(" ")
            processed_prefix_sum_list = processed_prefix_sum_str.split(" ")

        # must do lower
        processed_doc_list = [[x.lower() for x in sent] for sent in processed_doc_list]
        processed_ref_sum_list = [x.lower() for x in processed_ref_sum_list]
        processed_prefix_sum_list = [x.lower() for x in processed_prefix_sum_list]

        # if self.rm_stop_word:
        #     processed_doc_list = [[x for x in sent if x not in self.stop_words] for sent in processed_doc_list]
        #     processed_ref_sum_list = [x for x in processed_ref_sum_list if x not in self.stop_words]
        #     processed_prefix_sum_list = [x for x in processed_prefix_sum_list if x not in self.stop_words]

        target_ref_sum_list = [x for x in processed_ref_sum_list if x not in processed_prefix_sum_list]

        # TODO
        f_score_list, score_matrix = self.iter_rouge(processed_doc_list, target_ref_sum_list)

        # preprocessing finished
        filtered_doc_list, map_from_new_to_ori_idx = self.pre_prune(processed_doc_list, target_ref_sum_list)
        combination_data_dict = {}
        for num_sent_in_combination in range(self.min_combination_num, self.max_combination_num):
            combination_data = self.comp_num_seg_out_of_p_sent_beam(_filtered_doc_list=filtered_doc_list,
                                                                    num_sent_in_combination=num_sent_in_combination,
                                                                    target_ref_sum_list=target_ref_sum_list,
                                                                    map_from_new_to_ori_idx=map_from_new_to_ori_idx)
            if combination_data['best'] is None:
                break
            best_rouge_of_this_batch = combination_data['best']['R1']
            if len(combination_data_dict) >= self.beam_sz:
                rouge_in_bag = [float(k) for k, v in combination_data_dict.items()]
                if best_rouge_of_this_batch < min(rouge_in_bag):
                    break

            combination_data_dict = {**combination_data_dict, **combination_data['data']}
            combination_data_dict = collections.OrderedDict(sorted(combination_data_dict.items(), reverse=True))
            sliced = islice(combination_data_dict.items(), self.beam_sz)
            combination_data_dict = collections.OrderedDict(sliced)
            # combination_data_dict[num_sent_in_combination] = combination_data

        # prepare return data
        return_dict = {}
        for k, v in combination_data_dict.items():
            # tmp_list = [0 for _ in range(len_of_doc)]
            # for i in v['label']:
            #     tmp_list[i] = 1
            return_dict[k] = v['label']
        return return_dict

    def iter_rouge(self, list_of_doc, ref_sum):
        f_score_list = [self.get_rouge_ready_to_use(ref_sum, x) for x in list_of_doc]
        # score_matrix_delta = [[0 for _ in range(len(list_of_doc))] for _ in range(len(list_of_doc))]
        score_matrix = [[0 for _ in range(len(list_of_doc))] for _ in range(len(list_of_doc))]
        input = []
        for idx, x in enumerate(list_of_doc):
            for jdx, y in enumerate(list_of_doc):
                input.append((idx, jdx, ref_sum,x+y))
                s = self.get_rouge_ready_to_use(ref_sum, x + y)
                score_matrix[idx][jdx] = s
                # if f_score_list[idx] < 0.01:
                #
                #     score_matrix_delta[idx][jdx] = 0
                # else:
                #     score_matrix_delta[idx][jdx] = min(s / (f_score_list[idx] + 0.001), 2)
        # import numpy as np
        # np.set_printoptions(precision=2)
        # import seaborn as sns
        # sns.set()
        # f_score_list = np.asarray([f_score_list, f_score_list])
        # bx = sns.heatmap(f_score_list)
        # fig = bx.get_figure()
        # fig.savefig("individual_output.png")
        # print('-' * 30)
        # print(np.asarray(score_matrix))
        # score_matrix_delta = np.asarray(score_matrix_delta)
        # ax = sns.heatmap(score_matrix_delta)
        # fig = ax.get_figure()
        # fig.savefig("output.png")


        # ncpu=multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(processes=ncpu)
        # results = pool.starmap(self.get_rouge_ready_to_use, input)
        # for r in results:
        #     score, idx,jdx = r
        #     score_matrix[idx][jdx] = score
        return f_score_list, score_matrix
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
                    average_f_score = self.get_rouge_ready_to_use(gold_tokens=target_ref_sum_list,
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
            f1 = self.get_rouge_ready_to_use(target_ref_sum_list, candidate_doc_list)

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

    @staticmethod
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def get_rouge_ready_to_use_w_index(self,gold_tokens: List[str],
                               pred_tokens: List[str],idx,jdx):
        return self.get_rouge_ready_to_use(gold_tokens,pred_tokens), idx,jdx
    # No synomous standard version

    def get_rouge_ready_to_use(self, gold_tokens: List[str],
                               pred_tokens: List[str]):
        len_gold = len(gold_tokens)
        len_pred = len(pred_tokens)

        gold_bigram = _get_ngrams(2, gold_tokens)
        pred_bigram = _get_ngrams(2, pred_tokens)

        if self.rm_stop_word:
            gold_unigram = set([x for x in gold_tokens if x not in self.stop_words])
            pred_unigram = set([x for x in pred_tokens if x not in self.stop_words])
        else:
            gold_unigram = set(gold_tokens)
            pred_unigram = set(pred_tokens)

        rouge_1 = cal_rouge(pred_unigram, gold_unigram, len_pred, len_gold)['f']
        rouge_2 = cal_rouge(pred_bigram, gold_bigram, len_pred, len_gold)['f']
        rouge_score = (rouge_1 + rouge_2 ) / 2
        return rouge_score

    def pre_prune(self, list_of_doc: List[List[str]],
                  ref_sum: List[str]
                  ):
        keep_candidate_num = math.ceil(len(list_of_doc) * self.prune_candidate_percent)
        # f_score_list = [self.get_approximate_rouge(ref_sum, x) for x in list_of_doc]
        f_score_list = [self.get_rouge_ready_to_use(ref_sum, x) for x in list_of_doc]
        top_p_sent_idx = numpy.argsort(f_score_list)[-keep_candidate_num:]

        map_from_new_to_ori_idx = []
        # filter
        filtered_doc_list = []
        for i in range(len(top_p_sent_idx)):
            filtered_doc_list.append(list_of_doc[top_p_sent_idx[i]])
            map_from_new_to_ori_idx.append(top_p_sent_idx[i])
        return filtered_doc_list, map_from_new_to_ori_idx


if __name__ == '__main__':
    d = DocumentOracleDerivation(mixed_combination=True, tokenization=False)
    with open('example.txt', 'r') as fd:
        lines = fd.read().splitlines()
    ref_sum = [
        "Crunch vote on Mrs von der Leyen's appointment taking place this afternoon",
        "Mrs von der Leyen needs the support of a majority of MEPs to be EU's new boss",
        "She will be first female president of the European Commission if she wins vote",
        "She is set to replace Jean - Claude Juncker on November 1 - the day after Brexit",
        "Mrs von der Leyen said she is 'ready' to agree a new Brexit delay past October 31",
        "Nigel Farage said the comments had 'just made Brexit a lot more popular'",
        "Boris Johnson last night recommitted to his 'do or die' October 31 Brexit promise",
        "He and Jeremy Hunt also said backstop must be deleted to agree new deal",
    ]

    output = d.derive_doc_oracle(lines, " ".join(ref_sum))
    print(output)
