import torch, random

from data_preparation.search_algo import appx_simple_rouge_estimator
from model.data_reader import CNNDMDatasetReader
from model.decoding_util import universal_decoding_interface
from model.pyrouge_metrics import PyrougeEvaluation
from model.sem_red_map import MapKiosk
import numpy as np

if __name__ == '__main__':
    # sample some sentences, see if
    # trigram blocking
    # with ref sem red
    # without ref sem red
    # with ref+oracle sem red
    # works
    tmp_dir = '/datadrive/tmp'
    prg = PyrougeEvaluation(name='rouge_test',
                            cand_path=tmp_dir,
                            ref_path=tmp_dir,
                            path_to_valid=tmp_dir
                            )
    prg_disco = PyrougeEvaluation(name='rouge_disco',
                                  cand_path=tmp_dir,
                                  ref_path=tmp_dir,
                                  path_to_valid=tmp_dir
                                  )

    dataset = []
    import os

    d_name = 'cnndm'
    files = os.listdir('/datadrive/data/{}/test/'.format(d_name))
    for f in files:
        _d = torch.load('/datadrive/data/{}/test/'.format(d_name) + f)
        dataset += _d
    # random.shuffle(dataset)
    dataset = dataset[:100]
    # datasetreader = CNNDMDatasetReader()
    for d in dataset:

        lab = d['labels'][0]
        d_lab = d['d_labels'][0]
        # print(d)
        sent = d['sent_txt']
        sents = [" ".join(s) for idx, s in enumerate(sent) if lab[idx] == 1]
        disco_txt = d['disco_txt']
        discos = [" ".join(s) for idx, s in enumerate(disco_txt) if d_lab[idx] == 1]
        # tgt = d['tgt_tok_list_list_str']
        tgt_str = d['tgt_list_str']
        print(tgt_str)
        # uni_f_score_list = [appx_simple_rouge_estimator(
        #     sent[i], tgt
        # ) + random.random() / 5 for i in range(len(sent))]
        # maps = map_service.single_entry_entrance(sent, tgt
        #                                          )
        # sem_red_map = maps['red_map_p']
        # # sem_red_map = maps['red_map_f']
        #
        # disco_map_to_sent = datasetreader.map_disco_to_sent(d['d_span'])
        # pred_word_lists = universal_decoding_interface(np.asarray(uni_f_score_list),
        #                                                sem_red_map,
        #                                                use_disco=False,
        #                                                source_txt=sent,
        #                                                dependency_dict=None,
        #                                                trigram_block=trigram_block,
        #                                                max_pred_unit=3,
        #                                                disco_map_2_sent=disco_map_to_sent,
        #                                                threshold=0.3)
        # prg("<q>".join(sents), "<q>".join(tgt_str), "", "")
        # prg_disco("<q>".join(discos), "<q>".join(tgt_str), "", "")

    # prg.get_metric(True)
    # prg_disco.get_metric(True)
