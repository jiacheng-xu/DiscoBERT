# preprocess the multi_news dataset
"""
replace NEWLINE_CHAR

budget of total length = 800 tokens  (inflate to around 1k after bert bpe?)
so it will be 800/S

    split sources into different files

test.src  test.tgt  train.src  train.tgt  val.src  val.tgt
"""
DOC_SEG_TOKEN = '|||||'
LINE_SEG_TOKEN = 'NEWLINE_CHAR'
START_OF_TGT = '– '
DIR = '/datadrive/data/multi-news-original'
name = ['train', 'val', 'test']
suffix = ['src', 'tgt']
DOC_dir = 'raw_doc'
SUM_dir = 'sum'

# file_name.story.sum
# 001fb4ca3bd3a0c1cd91fdc813f0ebeeac678e76.story.doc
# raw_doc sum
import os
BUDGET = 1000

import random
def process_one_document(source: str):
    splits = source.split(DOC_SEG_TOKEN)
    random.shuffle(splits)
    bag = []
    candidate_sentences = []
    num_of_splits = len(splits)
    for sp in splits:
        lines = sp.split(LINE_SEG_TOKEN)
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if len(l) > 5]
        lines = [l for l in lines if len(l.split(' ')) > 2]
        candidate_sentences.append(lines)
    print()
def read_doc(dir, source_file, name):
    with open(os.path.join(dir, source_file), 'r') as fd:
        lines = fd.read().splitlines()
    print(lines[0])
    for l in lines:
        process_one_document(l)


def read_sum(dir, target_file):
    pass


if __name__ == '__main__':
    read_doc(DIR, name[1] + '.src', name[1])
    read_sum(DIR, name[1] + '.tgt')
