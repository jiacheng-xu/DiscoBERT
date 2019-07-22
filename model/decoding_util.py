def resolve_dependency(dep):
    dic = {}
    for _d in dep:
        source_node, tgt_node = _d
        if source_node == tgt_node:
            continue
        if source_node - 1 in dic:
            dic[source_node - 1] = list(set(dic[source_node - 1] + [tgt_node - 1]))
        else:
            dic[source_node - 1] = [tgt_node - 1]
    return dic


from collections import deque
from typing import List
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _decode_disco(sel_index, dep_dict) -> List[int]:
    q = deque()
    candidate_list = []
    q.append(sel_index)
    while len(q) > 0:
        ele = q.popleft()
        candidate_list.append(ele)
        if ele in dep_dict:
            [q.append(x) for x in dep_dict[ele]
             if (x not in candidate_list)]
    candidate_list = list(set(candidate_list))

    return candidate_list


def _decode_sent(sel_i):
    return [sel_i]


flatten = lambda l: [item for sublist in l for item in sublist]
# from model.tensor_bert import flatten, extract_n_grams, easy_post_processing
from model.model_util import extract_n_grams, easy_post_processing, split_sentence_according_to_id


def std_decode(sel_indexes, use_disco, source_txt, dependency_dict,
               trigram_block, min_pred_word, max_pred_word, step):
    pred_word_lists = [[] for i in range(int((max_pred_word - min_pred_word) / step))]
    pred_indexes_lists = [[] for i in range(int((max_pred_word - min_pred_word) / step))]
    num_slots = len(pred_word_lists)

    trigrams = set()
    current_words = []
    current_indexes = []
    hoop_cnt = 0
    for sel_i in sel_indexes:
        hoop_cnt += 1
        try:
            if use_disco:
                candidates = _decode_disco(sel_i, dependency_dict)
            else:
                candidates = _decode_sent(sel_i)

            if not trigram_block:
                new_candidate = list(set(candidates).union(set(current_indexes)))
                new_candidate.sort()
            else:
                non_overlap_candidate = []
                new_candidate = list(set(candidates).difference(set(current_indexes)))
                new_candidate.sort()
                for cand in new_candidate:
                    cand_trigram = extract_n_grams(" ".join(source_txt[cand]))
                    if trigrams.isdisjoint(cand_trigram):
                        non_overlap_candidate.append(cand)
                        trigrams.update(cand_trigram)
                    # else:
                    #     print("traigram blocked")
                    #     print("Candiate:{}".format(cand_trigram))
                    #     print("Exsit:{}".format(trigrams))
                all_candidate = list(set(non_overlap_candidate).union(set(current_indexes)))
                all_candidate.sort()
                new_candidate = all_candidate
            new_words = flatten([source_txt[c] for c in new_candidate])
            slot_to_go = int((len(new_words) - min_pred_word) / step)
            if slot_to_go >= num_slots or hoop_cnt > 20:
                break
            else:
                pred_indexes_lists[slot_to_go] = new_candidate
                pred_word_lists[slot_to_go] = new_words
                current_indexes = new_candidate
        except IndexError:
            logger.warning("Index Error\n{}".format(source_txt))

    # backup
    for idx, pred_word in enumerate(pred_word_lists):
        if pred_word == [] and idx > 0:
            pred_word_lists[idx] = pred_word_lists[idx - 1]
            pred_indexes_lists[idx] = pred_indexes_lists[idx - 1]
        elif pred_word == [] and idx < len(pred_word_lists)-1:
            pred_word_lists[idx] = pred_word_lists[idx +1]
            pred_indexes_lists[idx] = pred_indexes_lists[idx + 1]
    pred_word_strs_list = [[] for _ in range(num_slots)]
    # split sentences
    for idx, pred in enumerate(pred_indexes_lists):
        splited = split_sentence_according_to_id(pred)
        _t = []
        for sp in splited:
            x = flatten([source_txt[s] for s in sp])
            _t.append(" ".join(easy_post_processing(x)))
        pred_word_strs_list[idx] = _t
    # for idx, pred_word in enumerate(pred_word_lists):
    #     pred_word_strs.append(
    #         " ".join(easy_post_processing(pred_word))
    #     )
    return pred_word_strs_list


def pivot_decode():
    pass


def pivot_supervision():
    pass


import numpy as np


def decode_entrance(prob, meta_data, use_disco, trigram_block: bool = True,
                    use_pivot_decode: bool = False,
                    min_pred_word: int = 40, max_pred_word: int = 80,
                    step: int = 10
                    ):
    tgt = meta_data['tgt_txt']
    sel_indexes = np.argsort(-prob)
    if use_disco:
        src = meta_data['disco_txt']
        dep = meta_data['disco_dep']
        dep_dic = resolve_dependency(dep)
    else:
        src = meta_data['sent_txt']
        dep, dep_dic = None, None
    if use_pivot_decode:
        pred_word_strs = pivot_decode()
    else:
        pred_word_strs = std_decode(sel_indexes, use_disco, src, dep_dic,
                                    trigram_block, min_pred_word, max_pred_word, step)
    return pred_word_strs, tgt
