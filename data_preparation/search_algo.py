import re, itertools

from data_preparation.nlpyang_utils import _get_word_ngrams, _get_ngrams


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


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    from data_preparation.nlpyang_data_builder import cal_rouge
    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


from typing import List


def appx_simple_rouge_estimator(sent: List[str], abs: List[List[str]]):
    abstract: List[str] = sum(abs, [])
    eval_len = len(sent)
    ref_len = len(abstract)
    evaluated_1grams = _get_word_ngrams(1, [sent])
    reference_1grams = _get_word_ngrams(1, [abstract])
    # from data_preparation.nlpyang_data_builder import cal_rouge
    evaluated_2grams = _get_word_ngrams(2, [sent])
    reference_2grams = _get_word_ngrams(2, [abstract])
    # rouge_1 = cal_rouge(evaluated_1grams, reference_1grams)['f']
    rouge_1 = dedup_cal_rouge(evaluated_1grams, reference_1grams, eval_len, ref_len)
    rouge_1_f = rouge_1['f']
    rouge_1_r = rouge_1['r']
    rouge_1_p = rouge_1['p']
    rouge_2 = dedup_cal_rouge(evaluated_2grams, reference_2grams, eval_len, ref_len)
    rouge_2_f = rouge_2['f']
    rouge_2_r = rouge_2['r']
    rouge_2_p = rouge_2['p']
    # rouge_2 = cal_rouge(evaluated_2grams, reference_2grams)['f']
    # return rouge_1_f, rouge_1_r, rouge_1_p, rouge_2_f, rouge_2_r, rouge_2_p
    return rouge_1_p + rouge_2_p


def original_greedy_selection(doc_sent_list: List[List[str]], abstract_sent_list: List[List[str]], summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    from data_preparation.nlpyang_data_builder import cal_rouge
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)
