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
from model.decoding_util import fill_upper_right_matrix

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

    def get_red_mag_supervision(self, original_red_map, label_sensitive=0.5):

        optimal_indexes, optimal_values = self.margin_label_translator(
            original_red_map)  # get the optimal label for each row
        l = optimal_values.shape[0]

        row_optimal_va = optimal_values.reshape([l, 1])
        row_broadcasted_optimal_va = np.tile(row_optimal_va, l)
        row_red_map_supervision_label_mask = ((row_broadcasted_optimal_va - original_red_map) > label_sensitive)

        col_optimal_va = optimal_values.reshape([1, l])
        col_broadcasted_optimal_va = np.tile(col_optimal_va, [l, 1])
        col_red_map_supervision_label_mask = ((col_broadcasted_optimal_va - original_red_map) > label_sensitive)
        red_map_supervision_label_mask = np.logical_or(row_red_map_supervision_label_mask,
                                                       col_red_map_supervision_label_mask)
        # red_map_supervision_label_mask = ((broadcasted_optimal_va - original_red_map) > label_sensitive).astype(int)

        original_red_map_mask = (original_red_map > 0)
        final_mask = np.logical_and(red_map_supervision_label_mask, original_red_map_mask).astype(int)
        return final_mask, optimal_indexes

    @staticmethod
    def pick_label(redundancy_map, percentile=10):
        mask = redundancy_map > 0
        flatten_redundancy_map = redundancy_map.flatten()
        flatten_redundancy_map = flatten_redundancy_map[flatten_redundancy_map > 0]
        lower = np.percentile(flatten_redundancy_map, percentile)
        upper = np.percentile(flatten_redundancy_map, 100 - percentile)

        pos_map = np.logical_and(mask, redundancy_map > upper)
        neg_map = np.logical_and(mask, redundancy_map < lower)
        return pos_map, neg_map

    def single_entry_entrance(self, sentences: List[List[str]], abstract: List[List[str]]):
        abstract: List[str] = sum(abstract + sentences[:3], [])
        # abstract: List[str] = sum(abstract , [])
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

        # word overlapping heuristic
        unigram_overlap = [[0.0 for _ in range(len(sentences))] for _ in range(len(sentences))]

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

                if len(comb_1gram) > 0:
                    _over = len(eval_1grams_row.intersection(evaluated_1grams[col])) / len(
                        comb_1gram)
                    unigram_overlap[row][col] = _over
                    unigram_overlap[col][row] = _over
        # redundancy map. empty diag line
        red_map_f = self.get_redundancy_map(sal_map_f)
        red_map_p = self.get_redundancy_map(sal_map_p)  # exclude diag  # get original redundancy map
        # diag == -1. invalid pos == -1

        red_p_pos, red_p_neg = self.pick_label(red_map_p)
        red_f_pos, red_f_neg = self.pick_label(red_map_f)
        # red_p_supervision_mask, red_map_p_optimal_index = self.get_red_mag_supervision(red_map_p)
        # label_bin_red_map_p = self.binary_label_translator(red_map_f)

        unigram_overlap = np.asarray(unigram_overlap, dtype=np.float32)

        # label_bin_red_map_p = self.binary_label_translator(red_map_p)
        # label_bin_sal_map_f = self.binary_label_translator(sal_map_f, True)
        # label_bin_sal_map_p = self.binary_label_translator(sal_map_p, True)
        rt_dict = {
            'unigram_overlap': unigram_overlap,  # exclude diag
            # mar maps
            # 'red_map_p_supervision_mask': red_map_p_supervision_mask,  # where
            # 'red_map_p_opt_idx': red_map_p_optimal_index,
            'red_p_pos': red_p_pos,
            'red_p_neg': red_p_neg,
            'red_f_pos': red_f_pos,
            'red_f_neg': red_f_neg,
            # 'red_map_p': red_map_p,
            # 'red_map_f': red_map_f,
            # 'bin_red_map_f': label_bin_red_map_f,
            # 'bin_red_map_p': label_bin_red_map_p,
            # 'bin_sal_map_f': label_bin_sal_map_f,
            # 'bin_sal_map_p': label_bin_sal_map_p
        }
        return rt_dict

    @staticmethod
    def margin_label_translator(input_map: np.ndarray, check_diag: bool = False):

        valid_len = input_map.shape[0]
        # np_input_map = np.asarray(input_map)
        if not check_diag:
            input_map[range(valid_len), range(valid_len)] = -1
        agmax = np.argmax(input_map, axis=1)
        max_values = input_map[range(valid_len), agmax]
        return agmax, max_values

    @staticmethod
    def binary_label_translator(input_map: np.ndarray,
                                check_diag: bool = False,
                                percentile_based: bool = True,
                                percent: int = 15,
                                threshold_based: bool = False,
                                pos_thres: float = 1.0,
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
        elif not percentile_based and threshold_based:
            pass
        else:
            raise NotImplementedError

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
    def get_redundancy_map(sal_map: List[List[float]]) -> np.ndarray:
        """
        Get the redundancy map. Criteria: ROUGE[A,B]/max(R[A], R[B])
        """
        l = len(sal_map)
        # _data_insight = []
        redundancy_map = [[-1 for _ in range(l)] for _ in range(l)]
        for i in range(l):
            for j in range(i + 1):
                if i == j:
                    redundancy_map[i][j] = -1
                    continue
                max_of_individual = max(sal_map[i][i], sal_map[j][j])
                # min_of_individual = min(sal_map[i][i], sal_map[j][j])
                if max_of_individual > 0:  # and min_of_individual>0.1:
                    rr = sal_map[i][j] / max_of_individual
                    redundancy_map[i][j] = rr
                    redundancy_map[j][i] = rr
                # else:
                #     redundancy_map[i][j] = -1
                #     redundancy_map[i][j] = -1
                # _data_insight.append(sal_map[i][j] / max_of_individual)

        # print(statistics.mean(_data_insight))
        # print(statistics.stdev(_data_insight))
        # print(statistics.median(_data_insight))
        # redundancy_map = fill_upper_right_matrix(redundancy_map)
        redundancy_map = np.asarray(redundancy_map, dtype=np.float32)
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
