import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from os.path import join as pjoin

import torch
from multiprocess import Pool
from multiprocessing import pool
from pytorch_pretrained_bert import BertTokenizer

# from nlpyang_others_logging import logger
# from nlpyang_others_utils import clean
# from nlpyang_utils import _get_word_ngrams

from data_preparation.nlpyang_others_logging import logger
from data_preparation.nlpyang_others_utils import clean
from data_preparation.nlpyang_utils import _get_word_ngrams

"""
bf3dd673d72edf70f431bb3a638a3cc124a3c3ae.story.doc.merge 
7       1       The     the     DT      det     3       O        (ROOT (S (NP (DT The)  21
7       2       Supreme Supreme NNP     compound        3       ORGANIZATION     (NNP Supreme)  21
7       3       Court   Court   NNP     nsubj   5       ORGANIZATION     (NNP Court))   21
7       4       twice   twice   RB      advmod  5       O        (ADVP (RB twice))      21
7       5       ruled   rule    VBD     root    0       O        (VP (VBD ruled)        21

bf3dd673d72edf70f431bb3a638a3cc124a3c3ae.story.doc.brackets 
((9, 10), 'Nucleus', 'same_unit')
((11, 11), 'Nucleus', 'same_unit')
((9, 11), 'Satellite', 'manner')
((8, 11), 'Nucleus', 'span')
((12, 12), 'Satellite', 'attribution')
((13, 13), 'Nucleus', 'span')
((14, 14), 'Satellite', 'elaboration')
((13, 14), 'Nucleus', 'span')
((15, 15), 'Satellite', 'elaboration')
((13, 15), 'Nucleus', 'span')
((12, 15), 'Nucleus', 'span')
"""


def read_merge(file):
    with open(file, 'r') as fd:
        lines = fd.read().splitlines()
    # simply treat EDU as sentence
    source = []
    tmp = []
    tmp_edu_id = 1
    lines = [l for l in lines if len(l) > 0]
    for l in lines:
        items = l.split("\t")

        word = items[2].lower()
        edu_id = int(items[-1])
        sent_id = int(items[0])
        if edu_id == tmp_edu_id:
            tmp.append(word)
        else:
            source.append(" ".join(tmp))
            tmp = []
            tmp.append(word)
            tmp_edu_id = edu_id
    if tmp != []:
        source.append(" ".join(tmp))

    return [clean(sent).split(" ") for sent in source]


from collections import OrderedDict
from ast import literal_eval as make_tuple


def return_tree(d: OrderedDict):
    root_node = d.popitem()  # root node
    root_node_sidx, root_node_eidx, root_node_node, root_node_rel = root_node[1]
    if len(d) == 0:
        minimal_nucleus = [root_node_sidx] if root_node_node == 'Nucleus' else []
        return {
            'left': None,
            'right': None,
            'node': root_node[1],
            'minimal': minimal_nucleus,
            'dep': []
        }
    listed_items = list(d.items())
    right_child = listed_items[-1]  #
    r_sidx, r_eidx, r_node, r_rel = right_child[1]

    l_sidx = root_node_sidx
    l_eidx = r_sidx - 1

    keys = list(d.keys())
    cut_point = keys.index("{}_{}".format(l_sidx, l_eidx)) + 1
    left = listed_items[:cut_point]
    right = listed_items[cut_point:]

    left_node = return_tree(OrderedDict(left))
    right_node = return_tree(OrderedDict(right))
    my_minimal = []
    if left_node['node'][2] == 'Nucleus':
        my_minimal += left_node['minimal']
    if right_node['node'][2] == 'Nucleus':
        my_minimal += right_node['minimal']

    if left_node['left'] == None and left_node['node'][2] == 'Satellite':
        left_node['dep'] = my_minimal
    # elif left_node['left'] == None and left_node['node'][2] == 'Nucleus':
    if right_node['right'] == None and right_node['node'][2] == 'Satellite':
        right_node['dep'] = my_minimal
        # print(right_node['node'])
        # print(my_minimal)

    if right_node['right'] == None and left_node['right'] == None:
        unit = [left_node, right_node]
    elif right_node['right'] == None and left_node['right']:
        unit = left_node['unit'] + [right_node]
    elif right_node['right'] and left_node['right'] == None:
        unit = right_node['unit'] + [left_node]
    else:
        unit = left_node['unit'] + right_node['unit']
    return {'left': left_node,
            'right': right_node,
            'node': root_node[1],
            'minimal': my_minimal,
            'unit': unit}

    # return_tree right  sidx - ?
    # return_tree left     ? - eidx


def read_bracket(bracket_file):
    # print(bracket_file)
    with open(bracket_file, 'r') as fd:
        lines = fd.read().splitlines()
    treebank = [None for _ in range(1000)]
    d = OrderedDict()
    max_num = -1

    for l in lines:
        tup = make_tuple(l)
        index, node, relation = tup
        sidx, eidx = index
        d['{}_{}'.format(sidx, eidx)] = [sidx, eidx, node, relation]
        max_num = max(max_num, eidx)
        if sidx == eidx:
            treebank[sidx] = [node, relation, None]
    d['{}_{}'.format(1, max_num)] = [1, max_num, 'Nucleus', 'ROOT']

    # construct tree
    constructed_tree = return_tree(d)
    # anchor minimal dependence of all satelite
    unit = constructed_tree['unit']
    meta = [[u['node'][0], u['node'][2], u['node'][3], u['dep']] for u in unit]
    meta = sorted(meta, key=lambda tup: tup[0])
    return meta


import nltk


def load_sum(fname, path):
    f = os.path.join(path, fname + '.story.sum')
    with open(f) as fd:
        lines = fd.read().splitlines()

    lines = [nltk.word_tokenize(l) for l in lines if len(l) > 1]
    return lines


def load_merge_bracket(fname, path):
    merge_file = fname + '.story.doc.merge'
    brack_file = fname + '.story.doc.brackets'
    source = read_merge(os.path.join(path, merge_file))
    tree_dep = read_bracket(os.path.join(path, brack_file))
    return source


def load_json(p, lower):
    source = []
    tgt = []
    flag = False

    for sent in json.load(open(p))["sentences"]:
        tokens = [t["word"] for t in sent["tokens"]]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            continue
        if (flag):
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


import multiprocessing
import multiprocess


def load_json_EDU_coref():
    pass


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

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


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

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


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, min_src_ntokens, max_src_ntokens, min_nsents, max_nsents):
        # self.args = args
        self.min_src_ntokens = min_src_ntokens
        self.max_src_ntokens = max_src_ntokens
        self.min_nsents = min_nsents
        self.max_nsents = max_nsents
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src, tgt, oracle_ids):
        # src is List[sent_1_str, sent_2_str, ...]

        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.min_src_ntokens)]

        src = [src[i][:self.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.max_nsents]
        labels = labels[:self.max_nsents]

        if (len(src) < self.min_nsents):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(chunk_path, oracle_mode, oracle_sent_num, min_src_ntokens=5, max_src_ntokens=200, min_nsents=3,
                   max_nsents=100):
    datasets = ['train', 'valid', 'test']

    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(chunk_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append(
                (json_f, pjoin(chunk_path, real_name.replace('json', 'bert.pt')),
                 oracle_mode, oracle_sent_num, min_src_ntokens, max_src_ntokens, min_nsents, max_nsents
                 )
            )
        print(a_lst)
        cnt = multiprocessing.cpu_count()
        pool = Pool(cnt)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def tokenize(raw_path, save_path, snlp='/datadrive/stanford-corenlp-full-2018-10-05'):
    """tokenization of raw text
    args.raw_path contains raw files
        files must end with .story
    args.save_path is the target dir
    args.snlp is the path to SNLP. Exmaple: /datadrive/stanford-corenlp-full-2018-10-05
    """
    # raw_path = args.raw_path
    # save_path = args.save_path

    stories_dir = os.path.abspath(raw_path)
    tokenized_stories_dir = os.path.abspath(save_path)

    # if args.snlp:
    #     snlp = args.snlp
    # else:
    #     snlp = '/datadrive/stanford-corenlp-full-2018-10-05'
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    print("Stories like: {}".format(stories[12]))
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('doc')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', '-Xmx100g', '-cp', '{}/*'.format(snlp),
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit,pos,lemma,ner,parse,coref',
               '-threads', '45',
               '-ssplit.newlineIsSentenceBreak', 'two', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'xml', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    print(" ".join(command))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def _format_to_bert(params):
    # json_file, args, save_file = params
    read_json_file, wt_pt_file, oracle_mode, oracle_sent_num, min_src_ntokens, max_src_ntokens, min_nsents, max_nsents = params
    if (os.path.exists(wt_pt_file)):
        logger.info('Ignore %s' % wt_pt_file)
        return

    bert = BertData(min_src_ntokens, max_src_ntokens, min_nsents, max_nsents)

    logger.info('Processing %s' % read_json_file)
    jobs = json.load(open(read_json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        if (oracle_mode == 'greedy'):
            oracle_ids = greedy_selection(source, tgt, oracle_sent_num)
        elif (oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, oracle_sent_num)
        else:
            raise NotImplementedError
        b_data = bert.preprocess(source, tgt, oracle_ids)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens,
                       "labels": labels,
                       "segs": segments_ids,
                       'clss': cls_ids,
                       'src_txt': src_txt,
                       "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % wt_pt_file)
    torch.save(datasets, wt_pt_file)
    datasets = []
    gc.collect()


def format_to_lines(map_urls_path, seg_path, tok_path, shard_size, save_path, summary_path, data_name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Creating {}".format(save_path))
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(map_urls_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(seg_path, '*.merge')):
        # print("reading {}".format(f))
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(real_name)
        elif (real_name in corpus_mapping['test']):
            test_files.append(real_name)
        elif (real_name in corpus_mapping['train']):
            train_files.append(real_name)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, seg_path, tok_path, summary_path) for f in corpora[corpus_type]]

        # unparel test
        format_processed_data(a_lst[0])

        pool = Pool(multiprocessing.cpu_count())
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(format_processed_data, a_lst):
            dataset.append(d)
            if (len(dataset) > shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(data_name, corpus_type, p_ct)
                with open(os.path.join(save_path, pt_file), 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(data_name, corpus_type, p_ct)
            with open(os.path.join(save_path, pt_file), 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


import xml.etree.ElementTree as ET
from lxml import etree


def load_snlp(f, tok_path):
    # f = 'ff97bd78c563c09ea47e2a4a69c0bd53901012d9'
    full_file_name = os.path.join(tok_path, "{}.story.doc.xml".format(f))
    with open(full_file_name) as fobj:
        xmlstr = fobj.read()
    tree = etree.parse(full_file_name)
    root = tree.getroot()

    doc_id = root.findall("./document/docId")[0]
    corefrences = root.findall("./document/coreference")[0]
    sentences = root.findall("./document/sentences")[0]
    return_sentences_obj = []
    for sent in list(sentences):
        sent_id = int(sent.attrib['id']) - 1
        sent_words = []
        ele_in_sent = list(sent)
        toks = ele_in_sent[0]
        token_list = list(toks)
        for t in token_list:
            word = list(t)[0]
            sent_words.append(word.text)
        parse = ele_in_sent[1].text
        return_sentences_obj.append({'sent_id': sent_id,  # USE 0 based indexing
                                     'tokens': sent_words,
                                     'parse': parse,
                                     'corefs': [[] for _ in range(len(sent_words))]
                                     })
    coref_bag = []
    for coref in list(corefrences):
        mentions = list(coref)

        return_mentions = []
        efficient_mentions = []
        repre_sent_id = -1
        for m in mentions:
            mention_elements = list(m)
            sent_location = int(mention_elements[0].text) - 1
            head_location_in_sent = int(mention_elements[3].text) - 1
            text = mention_elements[4].text
            if m.get('representative'):
                represent = True
                repre_sent_id = sent_location
            else:
                represent = False
            efficient_mentions.append((sent_location, head_location_in_sent, represent))
            return_mentions.append({'sent_id': sent_location,
                                    'word_id': head_location_in_sent,
                                    'text': text,
                                    'rep': represent
                                    })

        # load coref mention
        for single_coref in efficient_mentions:
            single_coref_sent, single_coref_word, _ = single_coref
            return_sentences_obj[single_coref_sent]['corefs'][single_coref_word] = efficient_mentions
        coref_bag.append(return_mentions)

    return {
        'doc_id': doc_id,
        'coref': coref_bag,
        'sent': return_sentences_obj
    }


def format_processed_data(param):
    f, seg_path, tok_path, summary_path = param
    snlp_dict = load_snlp(f, tok_path)  # 'doc_id' 'coref' 'sent'
    source = load_merge_bracket(f, seg_path)
    target = load_sum(f, summary_path)
    return {'src': source, 'tgt': target,
            'sent': snlp_dict['sent'],
            'doc_id': snlp_dict['doc_id'],
            'coref': snlp_dict['coref']}


def _format_to_lines(f):
    # print(f)
    try:
        source, tgt = load_json(f, True)
    except:
        print("problem {}".format(f))
    # print("Pass")
    return {'src': source, 'tgt': tgt}
