import torch, os

from data_preparation.data_structure import MSBertData
from data_preparation.my_format_to_bert import DiscourseUnit, SentUnit
from data_preparation.nlpyang_others_utils import clean
import random

def merge_shuffle_pt(path):
    os.chdir(path)
    files = os.listdir(path)
    all_dataset = []
    for file in files:
        dataset = torch.load(os.path.join(file))
        all_dataset += dataset

    random.shuffle(all_dataset)
    name = "cnndm.train.{}.bert.pt"

if __name__ == '__main__':
    # dataset = torch.load(os.path.join('/datadrive/data/cnndm/test', 'dailymail.test.1.bert.pt'))
    # for d in dataset:
    #     print(d)
    import json

    # from nltk.tokenize.stanford import StanfordTokenizer
    from nltk.tokenize import TweetTokenizer
    import nltk
    tknzr = TweetTokenizer()

    s = "Good muffins cost's ,\" good\" $3.88\nin 26-year-old ,New York.  Please buy me\ntwo of them.\nThanks."
    print(nltk.word_tokenize(s))
    # s= "craze has taken off thanks to owners posting pictures on social media<q>initial idea was to give the pets a more eye-grabbing and clean-cut look<q>one dog salon worker in taipei has insisted that 'the dogs do n't mind '"
    print(tknzr.tokenize(s))
    exit()
    path = '/datadrive/data/cnndm/train'
    os.chdir(path)

    file = 'dailymail.train.141.bert.pt'
    dataset = torch.load(os.path.join(file))
    # jobs = json.load(open(os.path.join(path,file)))
    print(len(jobs))

    length_limit = 512
    jobs = jobs[:100]
    bert_data = MSBertData(5, 5, 5, 5)
    # x = jobs[0]
    for d in jobs:
        # disco_node = d['disco_node']
        # gen = get_next_node(disco_node)
        # if len(disco_node) == 0:
        #     print(d)
        #     continue
        disco_dep = d['disco_dep']

        # disco_graph_links = d['disco_graph_links']
        disco_links = d['disco_link']  #####

        span, tgt = d['disco_span'], d['tgt']
        sent, doc_id, coref = d['sent'], d['doc_id'], d['coref']

        # First of all, assemble data and  LENGTH truncation
        budget = 0
        disco_bag = []
        sent_bag = []
        original_disco_txt_list_of_str = []
        for idx in range(len(sent)):

            this_sent = sent[idx]
            this_disco = span[idx]
            this_tokens = this_sent['tokens']
            this_tokens = [clean(x.lower()) for x in this_tokens]
            this_coref = this_sent['corefs']
            original_word_len = len(this_tokens)

            tmp_disco_bag = []
            for disc in this_disco:

                # tree_node = next(gen)
                start, end = disc
                disc_piece = DiscourseUnit(len(disco_bag) + len(tmp_disco_bag), idx, rel_start=start, rel_end=end)

                # disc_piece.add_dep_info(tree_node)
                disc_piece.add_dep(disco_dep)
                for jdx in range(start, end + 1):
                    _toks = this_tokens[jdx]

                    disc_piece.add_word(_toks)

                    # look at word jdx, see if any coref mentions applied.
                    _cor = this_coref[jdx]
                    if _cor != []:
                        disc_piece.add_mention(jdx)  # add the orinigla index of the word in the sentence
                        for _c in _cor:
                            disc_piece.add_coref(_c)
                    # finish loading coref
                tmp_disco_bag.append(disc_piece)
                budget += disc_piece.get_bert_wp_length()
            budget += 2
            if budget > length_limit:
                break
            else:
                disco_bag += tmp_disco_bag
                original_disco_txt_list_of_str += [x.get_readable_words_as_list() for x in tmp_disco_bag]
                s = SentUnit(idx, this_tokens, [x.bert_word_pieces for x in tmp_disco_bag], tmp_disco_bag)
                sent_bag.append(s)

        effective_disco_number = len(disco_bag)
        # clean disco_graph_links
        disco_graph_links = [(tup[0] - 1, tup[1] - 1, tup[2]) for tup in disco_links if
                             (tup[0] <= effective_disco_number and tup[1] <= effective_disco_number)]

        disc_oracle_ids, disc_spans, disc_coref = bert_data.preprocess_disc(disco_bag, tgt)

        src_tok_index, sent_oracle_labels, segments_ids, \
        cls_ids, original_sent_txt_list_of_str, tgt_txt = bert_data.preprocess_sent(sent_bag, summary=tgt)
        # TO have: src_subtoken_idxs [for bert encoder], labels[sent level and discourse level],
        # segments_ids[for bert encoder],
        # cls_ids[for sent level],
        # span indexs [ for discourse level]
        # entity coref linking edge [ sent level and discourse level]
        # discourse linking edge [discourse level only]
        # src_txt, tgt_txt

        # provide two versions, one based on discourse, one without.
        # w. multiple oracle

        # prepare discourse data
        # oracle is computed based on discourse

        # prepare sent data

        b_data_dict = {"src": src_tok_index,
                       "labels": sent_oracle_labels,
                       "segs": segments_ids,
                       'clss': cls_ids,
                       'sent_txt': original_sent_txt_list_of_str,
                       'disco_txt': original_disco_txt_list_of_str,
                       "tgt_txt": tgt_txt,
                       'd_labels': disc_oracle_ids,
                       'd_span': disc_spans,
                       'd_coref': disc_coref,
                       'd_graph': disco_graph_links,
                       'disco_dep': disco_dep,
                       'doc_id': doc_id

                       }