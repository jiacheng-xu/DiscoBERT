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


def fill_upper_right_matrix(inp_mat, valid_len: int = None):
    if type(inp_mat) == list:
        inp_mat = np.asarray(inp_mat)
    if valid_len == None:
        valid_len = inp_mat.shape[0]

    flipped_inp_mat = np.rot90(np.fliplr(inp_mat))
    triu_mask = np.ones((valid_len, valid_len))
    iu1 = np.triu_indices(valid_len, 1)
    triu_mask[iu1] = 0
    final_output = inp_mat * triu_mask + (1 - triu_mask) * flipped_inp_mat
    return final_output


from nltk.tokenize.treebank import TreebankWordDetokenizer

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

def std_decode_unit(sel_indexes, use_disco, source_txt, dependency_dict,
                    trigram_block, max_pred_unit, disco_map_2_sent, src_full_sent):
    # pred_word_lists = [[] for i in range(int((max_pred_word - min_pred_word) / step))]
    # pred_indexes_lists = [[] for i in range(int((max_pred_word - min_pred_word) / step))]
    pred_word_lists = []
    pred_indexes_lists = []
    current_trigrams = set()
    current_words = []
    current_indexes: List[List[int]] = [[]]
    hoop_cnt = 0
    for sel_i in sel_indexes:
        hoop_cnt += 1
        nothing_changed = False
        try:
            if use_disco:
                candidates = _decode_disco(sel_i, dependency_dict)
            else:
                candidates = _decode_sent(sel_i)
            candidates.sort()

            cur_index = current_indexes[-1]
            # there is some overlapping
            if not set(candidates).isdisjoint(set(cur_index)):
                if set(candidates).issubset(set(cur_index)):
                    # nothing changed
                    nothing_changed = True
                else:
                    # there is some overlap
                    if trigram_block:
                        # # compare c in candidates with current traigram
                        # if add, update the current trigram
                        tmp = cur_index
                        _len = len(tmp)
                        for c in candidates:
                            if c in cur_index:
                                continue
                            c_trigram = extract_n_grams(" ".join(source_txt[c]))
                            if current_trigrams.isdisjoint(c_trigram):
                                # current_indexes[jdx] = list({c}.union(set(cur_index)))
                                if c not in tmp:
                                    tmp.append(c)
                                current_trigrams.update(c_trigram)
                        if len(tmp) > _len:
                            current_indexes.append(list(set(tmp).union(set(cur_index))))
                        else:
                            nothing_changed = True
                    else:
                        # don't consider trigram
                        current_indexes.append(list(set(candidates).union(set(cur_index))))
            else:  # there is NO overlapping
                if trigram_block:
                    # compare c in candidates with current traigram
                    # if add, update the current trigram
                    tmp = []
                    for c in candidates:
                        c_trigram = extract_n_grams(" ".join(source_txt[c]))
                        if current_trigrams.isdisjoint(c_trigram):
                            tmp.append(c)
                            current_trigrams.update(c_trigram)
                    if len(tmp) > 0:
                        current_indexes.append(list(set(tmp).union(set(cur_index))))
                    else:
                        nothing_changed = True
                else:
                    current_indexes.append(list(set(candidates).union(set(cur_index))))

            if hoop_cnt > 20:
                break
            if nothing_changed:
                continue

            # pred_indexes_lists.append(sum(current_indexes, []))
            if len(current_indexes) > max_pred_unit + 1:
                break
        except IndexError:
            logger.warning("Index Error\n{}".format(source_txt))

    current_indexes = current_indexes[1:]
    assert len(current_indexes) > 0
    # backup if pred_indexes_lists is not enough
    while len(current_indexes) < max_pred_unit:
        current_indexes.append(current_indexes[-1])

    # split sentences
    for idx, pred in enumerate(current_indexes):
        pred.sort()
        splited = split_sentence_according_to_id(pred, use_disco, disco_map_2_sent)
        _t = []
        for sp in splited:
            sp.sort()
            x = flatten([source_txt[s] for s in sp])
            _t.append(TreebankWordDetokenizer().detokenize(easy_post_processing(x)))
            # _t.append(" ".join(easy_post_processing(x)))
        pred_word_lists.append(_t)
    # for idx, pred_word in enumerate(pred_word_lists):
    #     pred_word_strs.append(
    #         " ".join(easy_post_processing(pred_word))
    #     )

    return pred_word_lists


def std_decode(stop_by_word_cnt, sel_indexes, use_disco, source_txt, dependency_dict,
               trigram_block, min_pred_word, max_pred_word, step, min_pred_unit, max_pred_unit):
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
        elif pred_word == [] and idx < len(pred_word_lists) - 1:
            pred_word_lists[idx] = pred_word_lists[idx + 1]
            pred_indexes_lists[idx] = pred_indexes_lists[idx + 1]
        elif pred_word == []:
            print("OOOps")
            print(idx)
            print(pred_indexes_lists)
            raise NotImplementedError
            # pred_indexes_lists[idx] = [0]
    pred_word_strs_list = [[] for _ in range(num_slots)]
    # split sentences
    for idx, pred in enumerate(pred_indexes_lists):
        splited = split_sentence_according_to_id(pred, use_disco)
        _t = []
        for sp in splited:
            x = flatten([source_txt[s] for s in sp])
            _t.append(TreebankWordDetokenizer().detokenize(easy_post_processing(x)))
            # _t.append(" ".join(easy_post_processing(x)))
        pred_word_strs_list[idx] = _t
    # for idx, pred_word in enumerate(pred_word_lists):
    #     pred_word_strs.append(
    #         " ".join(easy_post_processing(pred_word))
    #     )
    return pred_word_strs_list


import numpy as np


def universal_decoding_interface(scores,
                                 sem_red_map,
                                 use_disco: bool,
                                 source_txt: List[List[str]],
                                 dependency_dict,
                                 trigram_block: bool,
                                 max_pred_unit: int,
                                 disco_map_2_sent: List = None,
                                 threshold: float = 0.1
                                 ):
    pred_word_lists = []
    current_trigrams = set()
    current_indexes: List[List[int]] = [[]]
    hoop_cnt = 0
    valid_len = len(source_txt)
    scores = scores[:valid_len]
    if sem_red_map is not None:
        sem_red_map = sem_red_map[:valid_len, :valid_len]

    while True:

        hoop_cnt += 1
        if hoop_cnt > 20:
            break
        try:
            if len(current_indexes) <= 1:
                sel_i = np.argmax(scores)
                scores[sel_i] -= 10
                current_indexes.append([sel_i])
                continue
            sel_i = np.argmax(scores)  # this is the candidate

            if scores[sel_i] < 0.001:
                break
            scores[sel_i] -= 10
            if use_disco:
                candidates = _decode_disco(sel_i, dependency_dict)
            else:
                candidates = _decode_sent(sel_i)

            candidates.sort()
            cur_index = current_indexes[-1]

            if set(candidates).issubset(set(cur_index)):
                # nothing changed
                continue
            else:
                # there is some overlap
                if trigram_block:
                    newstuff, current_trigrams = search_trigram_blocking(candidates, cur_index,
                                                                         current_trigrams,
                                                                         source_txt)
                    if newstuff is not None:
                        current_indexes.append(newstuff)
                elif sem_red_map is not None:
                    newstuff = search_sem_red(candidates, cur_index, sem_red_map, threshold)
                    if newstuff is not None:
                        current_indexes.append(newstuff)
                else:
                    current_indexes.append(list(set(candidates).union(set(cur_index))))

            if len(current_indexes) > max_pred_unit + 1:
                break

        except IndexError:
            print("Index Error")
            logger.warning("Index Error")

    current_indexes = current_indexes[1:]
    assert len(current_indexes) > 0
    # backup if pred_indexes_lists is not enough
    while len(current_indexes) < max_pred_unit:
        current_indexes.append(current_indexes[-1])

    # split sentences
    for idx, pred in enumerate(current_indexes):
        pred.sort()
        splited = split_sentence_according_to_id(pred, use_disco, disco_map_2_sent)
        _t = []
        for sp in splited:
            sp.sort()
            x = flatten([source_txt[s] for s in sp])
            _t.append(TreebankWordDetokenizer().detokenize(easy_post_processing(x)))
            # _t.append(" ".join(easy_post_processing(x)))
        pred_word_lists.append(_t)
    return pred_word_lists


import copy


def search_sem_red(candidates, cur_index, sem_red_map, threshold):
    tmp = copy.deepcopy(cur_index)
    _len = len(tmp)
    sem_red_map[:, cur_index] = 10
    rows = sem_red_map[cur_index]
    out = np.amin(rows, axis=0)
    for c in candidates:
        min_c = out[c]
        if min_c >= threshold and c not in cur_index:
            tmp.append(c)
    if len(tmp) > _len:
        return list(set(tmp).union(set(cur_index)))
    else:
        return None


def search_trigram_blocking(candidates, cur_index, current_trigrams, source_txt):
    tmp = copy.deepcopy(cur_index)
    _len = len(tmp)
    for c in candidates:
        if c in cur_index:
            continue
        c_trigram = extract_n_grams(" ".join(source_txt[c]))
        if current_trigrams.isdisjoint(c_trigram):
            # current_indexes[jdx] = list({c}.union(set(cur_index)))
            if c not in tmp:
                tmp.append(c)
            current_trigrams.update(c_trigram)
    if len(tmp) > _len:
        # current_indexes.append(list(set(tmp).union(set(cur_index))))
        return list(set(tmp).union(set(cur_index))), current_trigrams
    else:
        return None, current_trigrams


def matrix_decode(sel_indexes: np.ndarray,
                  use_disco: bool,
                  source_txt: List[List[str]],
                  dependency_dict,
                  trigram_block: bool,
                  max_pred_unit: int,
                  disco_map_2_sent: List = None
                  ):
    # we also show with trigram and w/o trigram
    pred_word_lists = []
    current_trigrams = set()
    current_indexes: List[List[int]] = [[]]
    decoded_indexes: List[List[int]] = [[]]
    hoop_cnt = 0
    valid_len = len(source_txt)

    sel_indexes = sel_indexes[:valid_len, :valid_len]

    diag = sel_indexes.diagonal()  # diag is the diagnoal of the matrix , stands for single scores

    flipped_sel_indexes = np.rot90(np.fliplr(sel_indexes))
    triu_mask = np.ones((valid_len, valid_len))
    iu1 = np.triu_indices(valid_len, 1)
    triu_mask[iu1] = 0
    final_sel_indexes = sel_indexes * triu_mask + (1 - triu_mask) * flipped_sel_indexes

    while True:

        hoop_cnt += 1
        nothing_changed = False
        try:
            # determine the best candidate
            if len(current_indexes) <= 1:
                # INIT
                sel_i = np.argmax(diag)
                current_indexes.append([sel_i])
                decoded_indexes.append([sel_i])
                continue

            # look at the matrix
            decoded_idx = decoded_indexes[-1]
            # sum over all decoded_idx with weight of diag
            tmp = np.zeros((valid_len))
            for _d in decoded_idx:
                tmp += final_sel_indexes[_d] * diag[_d]
            tmp[decoded_idx] = -10
            sel_i = int(np.argsort(tmp)[::-1][0])
            if use_disco:
                candidates = _decode_disco(sel_i, dependency_dict)
            else:
                candidates = _decode_sent(sel_i)
            candidates.sort()

            cur_index = current_indexes[-1]
            # there is some overlapping
            if not set(candidates).isdisjoint(set(cur_index)):
                if set(candidates).issubset(set(cur_index)):
                    # nothing changed
                    nothing_changed = True
                else:
                    # there is some overlap
                    if trigram_block:
                        # # compare c in candidates with current traigram
                        # if add, update the current trigram
                        tmp = cur_index
                        _len = len(tmp)
                        for c in candidates:
                            if c in cur_index:
                                continue
                            c_trigram = extract_n_grams(" ".join(source_txt[c]))
                            if current_trigrams.isdisjoint(c_trigram):
                                # current_indexes[jdx] = list({c}.union(set(cur_index)))
                                if c not in tmp:
                                    tmp.append(c)
                                current_trigrams.update(c_trigram)
                        if len(tmp) > _len:
                            current_indexes.append(list(set(tmp).union(set(cur_index))))
                            decoded_indexes.append(list({sel_i}.union(set(decoded_idx))))
                        else:
                            nothing_changed = True
                    else:
                        # don't consider trigram
                        current_indexes.append(list(set(candidates).union(set(cur_index))))
                        decoded_indexes.append(list({sel_i}.union(set(decoded_idx))))
            else:  # there is NO overlapping
                if trigram_block:
                    # compare c in candidates with current traigram
                    # if add, update the current trigram
                    tmp = []
                    for c in candidates:
                        c_trigram = extract_n_grams(" ".join(source_txt[c]))
                        if current_trigrams.isdisjoint(c_trigram):
                            tmp.append(c)
                            current_trigrams.update(c_trigram)
                    if len(tmp) > 0:
                        current_indexes.append(list(set(tmp).union(set(cur_index))))
                        decoded_indexes.append(list({sel_i}.union(set(decoded_idx))))
                    else:
                        nothing_changed = True
                else:
                    current_indexes.append(list(set(candidates).union(set(cur_index))))
                    decoded_indexes.append(list({sel_i}.union(set(decoded_idx))))
            if hoop_cnt > 20:
                break
            if nothing_changed:
                continue
            if len(current_indexes) > max_pred_unit + 1:
                break

        except IndexError:
            print("Index Error")
            logger.warning("Index Error")

    current_indexes = current_indexes[1:]
    assert len(current_indexes) > 0
    # backup if pred_indexes_lists is not enough
    while len(current_indexes) < max_pred_unit:
        current_indexes.append(current_indexes[-1])

    # split sentences
    for idx, pred in enumerate(current_indexes):
        pred.sort()
        splited = split_sentence_according_to_id(pred, use_disco, disco_map_2_sent)
        _t = []
        for sp in splited:
            sp.sort()
            x = flatten([source_txt[s] for s in sp])
            _t.append(TreebankWordDetokenizer().detokenize(easy_post_processing(x)))
            # _t.append(" ".join(easy_post_processing(x)))
        pred_word_lists.append(_t)
    # for idx, pred_word in enumerate(pred_word_lists):
    #     pred_word_strs.append(
    #         " ".join(easy_post_processing(pred_word))
    #     )
    # print(current_indexes)
    return pred_word_lists


def decode_entrance(prob, prob_mat, meta_data, use_disco, trigram_block: bool = True,
                    sem_red_map: bool = False,
                    pair_oracle: bool = False,
                    stop_by_word_cnt: bool = True,
                    min_pred_word: int = 40, max_pred_word: int = 80,
                    step: int = 10, min_pred_unit: int = 3,
                    max_pred_unit: int = 6,
                    threshold=0.05
                    ):
    tgt = meta_data['tgt_txt']
    src_full_sent = meta_data['sent_txt']
    disco_map_2_sent = meta_data['disco_map_to_sent']
    if use_disco:
        src = meta_data['disco_txt']
        dep = meta_data['disco_dep']
        dep_dic = resolve_dependency(dep)
    else:
        src = meta_data['sent_txt']
        dep, dep_dic = None, None

    if sem_red_map:
        if stop_by_word_cnt:
            raise NotImplementedError
        else:

            pred_word_strs = universal_decoding_interface(
                prob, prob_mat, use_disco,
                src, dep_dic,
                trigram_block,
                max_pred_unit, disco_map_2_sent, threshold

            )
    else:
        sel_indexes = np.argsort(-prob)
        if stop_by_word_cnt:

            pred_word_strs = std_decode(stop_by_word_cnt, sel_indexes, use_disco, src, dep_dic,
                                        trigram_block, min_pred_word, max_pred_word,
                                        step, min_pred_unit, max_pred_unit)
        else:
            pred_word_strs = std_decode_unit(sel_indexes, use_disco, src, dep_dic,
                                             trigram_block,
                                             max_pred_unit,
                                             disco_map_2_sent,src_full_sent
                                             )
    return pred_word_strs, tgt
