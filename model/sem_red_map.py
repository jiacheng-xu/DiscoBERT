#
"""
Semantic Salience & Redundancy Map
Goal: trainable deduplication; unifying salience estimation and redundancy avoidance; provide a new pairwise perspective

Salience Map: emphasize on R[A+B]   (marginalization could reduce to R[X])
-----
If we already select sentence 0, among all R[0, ?], R[0,8 | ref] is the highest
Hinge loss: max(0, NN_score[0; 7, 8] - (*) NN[0,8] + (hat) NN[0,7] )
-----
un-structural version
binarization of the salience map as the supervision signal

Redundancy Map: emphasize on R[A+B] / max(R[A], R[B])


Core algorithm:
Readout: Works similarly as trigram blocking. We still sort the score according to the uni vector

getattr(self, f"self_attention_{i}")
{sal, red} x {bin, mag} x {sec, hig}

TODO Efficient score computing
TODO Set up key names
"""
from model.decoding_util import flatten
import numpy as np

"""
Given a document, MapKiosk could provide all of the semantic maps you need.
"""

from typing import List

from data_preparation.nlpyang_utils import _get_ngrams, _get_word_ngrams
from data_preparation.search_algo import dedup_cal_rouge

import statistics


class MapKiosk():

    def __init__(self, map_keys=None):
        self.keys = map_keys

    def single_entry_entrance(self, sentences: List[List[str]], abstract: List[List[str]]):
        abstract: List[str] = sum(abstract, [])
        # cache ngrams for efficient reuse
        evaluated_1grams = [_get_word_ngrams(1, [s]) for s in sentences]
        evaluated_2grams = [_get_word_ngrams(2, [s]) for s in sentences]
        evaluated_lens = [len(s) for s in sentences]

        reference_1gram = _get_word_ngrams(1, [abstract])
        reference_2gram = _get_word_ngrams(2, [abstract])
        ref_len = len(abstract)
        _dict = {}
        # uni rouge R[A]
        uni_rouge_1: List[dict] = [dedup_cal_rouge(eva_1gram, reference_1gram, evaluated_lens[idx], ref_len) for
                                   idx, eva_1gram in enumerate(evaluated_1grams)]
        uni_rouge_2: List[dict] = [dedup_cal_rouge(eva_2gram, reference_2gram, evaluated_lens[idx], ref_len) for
                                   idx, eva_2gram in
                                   enumerate(evaluated_2grams)]
        # map rouge R[A+B]
        sal_map_f = [[0 for _ in range(len(sentences))] for _ in range(len(sentences))]
        sal_map_p = [[0 for _ in range(len(sentences))] for _ in range(len(sentences))]

        for row, sent in enumerate(sentences):
            eval_1grams_row = evaluated_1grams[row]
            eval_2grams_row = evaluated_2grams[row]
            for col in range(row + 1):
                if row == col:
                    sal_map_f[row][col] = uni_rouge_1[row]['f'] + uni_rouge_2[row]['f']
                    sal_map_p[row][col] = uni_rouge_1[row]['p'] + uni_rouge_2[row]['p']
                    continue
                comb_1gram = eval_1grams_row.union(evaluated_1grams[col])
                comb_2gram = eval_2grams_row.union(evaluated_2grams[col])
                _1rouge = dedup_cal_rouge(comb_1gram, reference_1gram, evaluated_lens[row] + evaluated_lens[col],
                                          ref_len)
                _2rouge = dedup_cal_rouge(comb_2gram, reference_2gram, evaluated_lens[row] + evaluated_lens[col],
                                          ref_len)
                sal_map_f[row][col] = _1rouge['f'] + _2rouge['f']
                sal_map_p[row][col] = _1rouge['p'] + _2rouge['p']
                # sal_map[row][col] = (_1rouge, _2rouge)
        # map redundancy
        red_map_f = self.get_redundancy_map(sal_map_f)
        red_map_p = self.get_redundancy_map(sal_map_p)

        label_bin_red_map_f = self.binary_label_translator(red_map_f)
        label_bin_red_map_p = self.binary_label_translator(red_map_p)
        label_bin_sal_map_f = self.binary_label_translator(sal_map_f, True)
        label_bin_sal_map_p = self.binary_label_translator(sal_map_p, True)
        rt_dict = {'bin_red_map_f': label_bin_red_map_f,
                   'bin_red_map_p': label_bin_red_map_p,
                   'bin_sal_map_f': label_bin_sal_map_f,
                   'bin_sal_map_p': label_bin_sal_map_p
                   }
        return rt_dict

    @staticmethod
    def binary_label_translator(input_map: List, check_diag: bool = False, percentile_based: bool = True,
                                percent: int = 30,
                                threshold_based: bool = False, pos_thres: float = 1.0,
                                neg_thres=0.7):
        l = len(input_map)
        label_map = [[-1 for _ in range(l)] for _ in range(l)]
        if percentile_based and not threshold_based:
            flattened_map = flatten(input_map)
            flattened_map = [x for x in flattened_map if x > 0]
            up_p = np.percentile(flattened_map, percent)
            down_p = np.percentile(flattened_map, 100 - percent)
            for row in range(l):
                if check_diag:
                    rg = row + 1
                else:
                    rg = row
                for col in range(rg):
                    if input_map[row][col] >= down_p:
                        label_map[row][col] = 1
                    elif input_map[row][col] <= up_p:
                        label_map[row][col] = 0
            return label_map

    @staticmethod
    def dedup_cal_rouge(evaluated_ngrams: set, reference_ngrams: set, evaluated_len: int, reference_len: int):
        # reference_count = len(reference_ngrams)
        # evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_len == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_len

        if reference_len == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_len

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return {"f": f1_score, "p": precision, "r": recall}

    def get_salience_map(self, keys):
        """
        given a document with sent or disco as the basic unit, output two salience maps
        during computation, all of the ngrams are precomputed in __init__ function
        only need to compute a triangle and duplicate to a full tensor for convinience (diag)
        keys like ["f_20","p_30", ...] {f or p}_{percentile}
        :return: arbitray num of maps. sal_f_map and sal_p_map.
                int type (either 1 (pairwise oracle) or 0 or -1 (mask out))
        """
        pass

    @staticmethod
    def get_redundancy_map(sal_map: List[List[float]]):
        """

        :return: arbitrary num of maps.
                int type (either 1 (pos edge) or 0 (neg edge) or -1 (mask out))
        """
        l = len(sal_map)
        # _data_insight = []
        redundancy_map = [[-1 for _ in range(l)] for _ in range(l)]
        for i in range(l):
            for j in range(i + 1):
                if i == j:
                    redundancy_map[i][j] = 1
                    continue
                max_of_individual = max(sal_map[i][i], sal_map[j][j])
                if max_of_individual > 0:
                    redundancy_map[i][j] = sal_map[i][j] / max_of_individual
                    # _data_insight.append(sal_map[i][j] / max_of_individual)

        # print(statistics.mean(_data_insight))
        # print(statistics.stdev(_data_insight))
        # print(statistics.median(_data_insight))
        return redundancy_map

    def get_ngram_map(self):
        pass


import torch, os

if __name__ == '__main__':

    path = '/datadrive/data/cnndm/train'
    os.chdir(path)
    from data_preparation.search_algo import appx_simple_rouge_estimator

    file = 'dailymail.train.141.bert.pt'
    dataset = torch.load(os.path.join(file))
    import random

    random.shuffle(dataset)
    for d in dataset:
        print(d)
        sent = d['sent_txt'][:20]
        tgt = d['tgt_tok_list_list_str']
        # {sal, red} x {bin, mag} x {sec, hig}
        map_kiosk = MapKiosk(['sal_bin_sec', 'sal_mag_sec'])
        maps = map_kiosk.single_entry_entrance(sent, tgt)
