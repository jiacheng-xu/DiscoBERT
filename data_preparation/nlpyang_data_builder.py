import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
from multiprocessing.pool import Pool
from os.path import join as pjoin

import gc
import torch
from pytorch_pretrained_bert import BertTokenizer

from data_preparation.my_format_to_bert import MS_formate_to_bert
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


def read_discourse_merge(file):
    """
    Return a list of list of tuples
    [ sent_i [ tuples (start, end (inclusive)
    :param file:
    :return:
    """
    # file = '/datadrive/data/cnn/segs/01a4c31112f1ceeeebbe8dce862f14f6792c03a3.story.doc.merge'
    with open(file, 'r') as fd:
        lines = fd.read().splitlines()
    # simply treat EDU as sentence
    lines = [l for l in lines if len(l) > 0]

    edu_dict = OrderedDict()

    last_line = lines[-1].split("\t")
    total_sent_num = int(last_line[0]) + 1
    sent_with_edu_spans = [[] for _ in range(total_sent_num)]
    for l in lines:
        items = l.split("\t")
        edu_id = str(items[-1])
        sent_id = int(items[0])
        word_index_in_sent = int(items[1]) - 1  # WARN this is indexed from 1 rather than 0 !!!!

        if edu_id in edu_dict:
            _t = edu_dict[edu_id]
            _t[2] = word_index_in_sent
        else:
            edu_dict[edu_id] = [sent_id, word_index_in_sent, word_index_in_sent]
    for k, v in edu_dict.items():
        sent_id, start_idx, end_idx = v
        sent_with_edu_spans[sent_id].append((start_idx, end_idx))

    EDU_pool = {}
    for k, v in edu_dict.items():
        EDU_pool[k] = v[0]

    return sent_with_edu_spans, EDU_pool


from collections import OrderedDict
from ast import literal_eval as make_tuple


def has_child(node):
    if (not node['left']) and (not node['right']):
        return False
    else:
        return True


def return_tree(d: OrderedDict):
    root_node = d.popitem()  # root node
    root_node_sidx, root_node_eidx, root_node_node, root_node_rel = root_node[1]
    if len(d) == 0:
        minimal_nucleus = [root_node_sidx] if root_node_node == 'Nucleus' else []
        return {
            'left': None,
            'right': None,
            's': root_node_sidx,
            'e': root_node_eidx,
            'type': root_node_node,
            'rel': root_node_rel,
            'head': root_node_sidx,
            # 'node': root_node[1],
            'minimal': minimal_nucleus,
            'dep': -1,
            'link': [],
            'unit': []
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

    # for link
    # left head
    left_head = left_node['head']
    left_node_type = left_node['type']
    # right head
    right_head = right_node['head']
    right_node_type = right_node['type']

    if left_node_type == right_node_type:
        my_head = left_head
        my_dep = (left_head, right_head, root_node_rel)
    elif left_node_type == 'Nucleus':
        my_head = left_head
        my_dep = (left_head, right_head, root_node_rel)
    else:
        my_head = right_head
        my_dep = (right_head, left_head, root_node_rel)
    # aggregate links from left and right
    all_links = left_node['link'] + right_node['link'] + [my_dep]

    # my_minimal = []
    # if left_node['node'][2] == 'Nucleus':
    #     my_minimal += left_node['minimal']
    # if right_node['node'][2] == 'Nucleus':
    #     my_minimal += right_node['minimal']

    if (not has_child(left_node)) and left_node['type'] == 'Satellite':
        left_node['dep'] = my_head
    if (not has_child(right_node)) and right_node['type'] == 'Satellite':
        right_node['dep'] = my_head  # TODO for the dependency , i am not 100% sure

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

            's': root_node_sidx,
            'e': root_node_eidx,
            'type': root_node_node,
            'rel': root_node_rel,
            'head': my_head,
            # 'dep':,
            'link': all_links,
            'unit': unit}

    # return_tree right  sidx - ?
    # return_tree left     ? - eidx


def determine_head(left_node, right_node):
    if left_node['type'] == 'n':
        return left_node['head']
    elif right_node['type'] == 'n':
        return right_node['head']
    else:
        return left_node['head']


def new_return_tree(d, EDU_pool):
    root_node = d.popitem()  # root node
    root_node_sidx, root_node_eidx, root_node_node, root_node_rel = root_node[1]
    root_node_node = 'n' if root_node_node.startswith('N') else 's'
    if len(d) == 0:
        # reach the leaf node
        return {
            'left': None,
            'right': None,
            's': root_node_sidx,
            'e': root_node_eidx,
            'type': root_node_node,
            'rel': root_node_rel,
            'head': root_node_sidx,
            # 'node': root_node[1],
            # 'minimal': minimal_nucleus,
            'dep': [],
            'link': [],
            'unit': []
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

    left_node = new_return_tree(OrderedDict(left), EDU_pool)
    right_node = new_return_tree(OrderedDict(right), EDU_pool)

    # print('here')
    # my head is my left (if there are two N sons) N's head
    my_head = determine_head(left_node, right_node)

    deps = left_node['dep'] + right_node['dep']
    # If one of my son is S, then add to dependency bank where dep(S) -> my head
    if left_node['type'] == 's' and right_node['type'] == 's':
        raise NotImplementedError

    if left_node['type'] == 's':
        if EDU_pool['{}'.format(left_node['s'])] == EDU_pool['{}'.format(my_head)] == EDU_pool[
            '{}'.format(left_node['e'])]:
            deps.append((left_node['head'], my_head))
            # for x in range(left_node['s'], left_node['e'] + 1):
            #     deps.append((x, my_head))
    if right_node['type'] == 's':
        if EDU_pool['{}'.format(right_node['s'])] == EDU_pool['{}'.format(right_node['e'])] == EDU_pool[
            '{}'.format(my_head)]:
            deps.append((right_node['head'], my_head))
            # for x in range(right_node['s'], right_node['e'] + 1):
            #     deps.append((x, my_head))
    # link: link the head of my sons
    links = left_node['link'] + right_node['link']
    links.append((left_node['head'], right_node['head'], root_node_rel))

    return {
        'left': left_node,
        'right': right_node,
        's': root_node_sidx,
        'e': root_node_eidx,
        'type': root_node_node,
        'rel': root_node_rel,
        'head': my_head,
        # 'node': root_node[1],
        # 'minimal': minimal_nucleus,
        'dep': deps,
        'link': links,
        # 'unit': []
    }


def new_read_bracket(bracket_file, EDU_pool):
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
    x = new_return_tree(d, EDU_pool)
    # print(x['link'])
    # print(x['dep'])
    # exit()
    return x['link'], x['dep']


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

    # we only need two things here:
    # unit is all leaf nodes, which we are going to use the dependency info

    unit = constructed_tree['unit']
    meta = [[u['s'], u['type'], u['rel'], u['dep']] for u in
            unit]  # 'node'[0] is the node id  [2] is the type N/S
    # [3] is the relation   [4] is the dependency
    meta = sorted(meta, key=lambda tup: tup[0])

    # link contains all the graph links
    link = constructed_tree['link']  # [(src,tgt, rel), ....]
    return meta, link


import nltk


def load_sum(fname, path):
    f = os.path.join(path, fname + '.story.sum')
    with open(f) as fd:
        lines = fd.read().splitlines()

    lines = [nltk.word_tokenize(l) for l in lines if len(l) > 1]
    lower_lines = []
    for l in lines:
        lower_lines.append([x.lower() for x in l])
    return lower_lines


def load_merge_bracket(fname, path):
    merge_file = fname + '.story.doc.conll.merge'
    brack_file = fname + '.story.doc.conll.brackets'
    disco_seg, EDU_pool = read_discourse_merge(os.path.join(path, merge_file))

    link, dep = new_read_bracket(os.path.join(path, brack_file), EDU_pool)
    # node_meta, graph_links = read_bracket(os.path.join(path, brack_file))
    return disco_seg, dep, link


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


def load_json_EDU_coref():
    pass


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
                   max_nsents=100, length_limit=766):
    datasets = ['train', 'valid', 'test']

    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(chunk_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append(
                (json_f, pjoin(chunk_path, real_name.replace('json', 'bert.pt')),
                 oracle_mode, oracle_sent_num, min_src_ntokens, max_src_ntokens, min_nsents, max_nsents, length_limit
                 )
            )
        print(a_lst)
        import random
        random.shuffle(a_lst)
        # MS_formate_to_bert(a_lst[0])
        _p = Pool(multiprocessing.cpu_count())
        for d in _p.imap(MS_formate_to_bert, a_lst):
            pass

        _p.close()
        _p.join()


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
    # 000ce9139d5bf974c2a621226b6ed77900bfa498.story.doc

    tokenized_stories = os.listdir(tokenized_stories_dir)  # 0016b5687760b3da5dd328e9b77ddd2522724a5a.story.doc.xml
    exist_files = [".".join(f.split(".")[:-1]) for f in tokenized_stories if f.endswith('.xml')]
    print("{} out of {} files processed".format(len(exist_files), len(stories)))
    todo_files = list(set(stories) - set(exist_files))
    stories = todo_files
    import random
    random.shuffle(stories)
    print("Stories like: {}".format(stories[0]))
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('doc')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    import multiprocessing
    n_cpu = multiprocessing.cpu_count() - 3
    command = ['java', '-Xmx100g', '-cp', '{}/*'.format(snlp),
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit,pos,lemma,ner,parse,coref',
               '-threads', '{}'.format(n_cpu),
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
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
        elif (oracle_mode == 'beam'):
            oracle_ids = beam_selection(source, tgt, oracle_sent_num)

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
        if 'nyt' in map_urls_path:# if you want to add new key , here
            for line in open(pjoin(map_urls_path, 'mapping_' + corpus_type + '.txt')):
                temp.append(line.strip())
            print("Examples: {}".format(temp[0]))
        else:
            for line in open(pjoin(map_urls_path, 'mapping_' + corpus_type + '.txt')):
                temp.append(hashhex(line.strip()))
            print("Examples: {}".format(temp[0]))
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
    # print(len(corpora['train']))
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, seg_path, tok_path, summary_path) for f in corpora[corpus_type]]
        print(len(a_lst))
        # unparel test
        format_processed_data(a_lst[0])

        run_pool = Pool(multiprocessing.cpu_count())
        dataset = []
        p_ct = 0
        for d in run_pool.imap_unordered(format_processed_data, a_lst):
            dataset.append(d)
            if (len(dataset) > shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(data_name, corpus_type, p_ct)
                with open(os.path.join(save_path, pt_file), 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        run_pool.close()
        run_pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(data_name, corpus_type, p_ct)
            with open(os.path.join(save_path, pt_file), 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


from lxml import etree


def load_snlp(f, tok_path):
    # f = 'ff97bd78c563c09ea47e2a4a69c0bd53901012d9'
    full_file_name = os.path.join(tok_path, "{}.story.doc.xml".format(f))
    with open(full_file_name) as fobj:
        xmlstr = fobj.read()
    tree = etree.parse(full_file_name)
    root = tree.getroot()

    doc_id = root.findall("./document/docId")[0].text
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
            start_location_in_sent = int(mention_elements[1].text) - 1
            end_location_in_sent = int(mention_elements[2].text) - 1
            text = mention_elements[4].text
            if m.get('representative'):
                represent = True
                repre_sent_id = sent_location
            else:
                represent = False
            efficient_mentions.append((sent_location, head_location_in_sent, represent))
            return_mentions.append({'sent_id': sent_location,
                                    'word_id': head_location_in_sent,
                                    'start_id': start_location_in_sent,
                                    'end_id': end_location_in_sent,
                                    'text': text,
                                    'rep': represent
                                    })

        # load coref mention
        # efficient_mentions keeps track of ONE set of coref mentions.
        # for simplicity, we will write the coref info into the sent/word.
        # for example, sent 01 word 01 has a list contains the corefs.

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
    span, dep, link = load_merge_bracket(f, seg_path)
    target = load_sum(f, summary_path)
    return {'disco_span': span,
            'disco_dep': dep,
            'disco_link': link,
            'tgt': target,
            'sent': snlp_dict['sent'],
            'doc_id': snlp_dict['doc_id'],
            'coref': snlp_dict['coref']
            }


if __name__ == '__main__':
    load_merge_bracket(
        'bf10dc849e5fa7325df269c2b28fcefebb5c956c', '/datadrive/data/cnn/segs/'
    )
