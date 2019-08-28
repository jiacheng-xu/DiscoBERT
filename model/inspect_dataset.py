import tempfile
from typing import List
from nltk.tokenize.treebank import TreebankWordDetokenizer

f = lambda x: TreebankWordDetokenizer().detokenize(x)
from model.pyrouge_metrics import test_rouge

if __name__ == '__main__':
    import torch

    # x = f(["I","love",'you'])

    tokenzied_lead3 = []
    detokenzied_lead3 = []
    tokenized_ref = []
    detokenized_ref = []
    # x = torch.load('/datadrive/GETSum/bert_data/cnndm.test.0.bert.pt')
    x = torch.load('/datadrive/data/cnndm/train/cnn.train.14.bert.pt')
    for every in x:
        lead3 = every['src_txt'][:3]
        tokenzied_lead3.append("<q>".join(lead3))
        detok_lead3 = "<q>".join([f(x.split()) for x in lead3])
        detokenzied_lead3.append(detok_lead3)
        tgt = every['tgt_txt']
        tgt_list_str = tgt.split("<q>")
        tgt_list_str = [f(x.split()) for x in tgt_list_str]

        tokenized_ref.append(tgt)
        detokenized_ref.append("<q>".join(tgt_list_str))
    cand_tok = '/datadrive/tmp/cand_tok.txt'
    with open(cand_tok, 'w') as fd:
        fd.write("\n".join(tokenzied_lead3))
    cand_detok = '/datadrive/tmp/cand_detok.txt'
    with open(cand_detok, 'w') as fd:
        fd.write("\n".join(detokenzied_lead3))
    ref_tok = '/datadrive/tmp/ref_tok.txt'
    with open(ref_tok, 'w') as fd:
        fd.write("\n".join(tokenized_ref))
    ref_detok = '/datadrive/tmp/ref_detok.txt'
    with open(ref_detok, 'w') as fd:
        fd.write("\n".join(detokenized_ref))

    p = tempfile.mkdtemp(prefix='/datadrive/tmp/')
    x = test_rouge(p, cand_tok, ref_detok)
    print(x)
    # {'rouge_1_recall': 0.53099, 'rouge_1_recall_cb': 0.52352, 'rouge_1_recall_ce': 0.53829, 'rouge_1_precision': 0.34368, 'rouge_1_precision_cb': 0.33844, 'rouge_1_precision_ce': 0.34934, 'rouge_1_f_score': 0.40349, 'rouge_1_f_score_cb': 0.3984, 'rouge_1_f_score_ce': 0.40877, 'rouge_2_recall': 0.23281, 'rouge_2_recall_cb': 0.22563, 'rouge_2_recall_ce': 0.23992, 'rouge_2_precision': 0.14862, 'rouge_2_precision_cb': 0.14399, 'rouge_2_precision_ce': 0.15356, 'rouge_2_f_score': 0.17541, 'rouge_2_f_score_cb': 0.1702, 'rouge_2_f_score_ce': 0.18096, 'rouge_l_recall': 0.48147, 'rouge_l_recall_cb': 0.47408, 'rouge_l_recall_ce': 0.48894, 'rouge_l_precision': 0.31195, 'rouge_l_precision_cb': 0.30694, 'rouge_l_precision_ce': 0.3175, 'rouge_l_f_score': 0.3661, 'rouge_l_f_score_cb': 0.36098, 'rouge_l_f_score_ce': 0.37154}
    # 'rouge_1_f_score': 0.40349

    p = tempfile.mkdtemp(prefix='/datadrive/tmp/')
    x = test_rouge(p, cand_detok, ref_tok)
    print(x)
    # {'rouge_1_recall': 0.53032, 'rouge_1_recall_cb': 0.52289, 'rouge_1_recall_ce': 0.53773, 'rouge_1_precision': 0.34407, 'rouge_1_precision_cb': 0.33883, 'rouge_1_precision_ce': 0.34975, 'rouge_1_f_score': 0.40353, 'rouge_1_f_score_cb': 0.39848, 'rouge_1_f_score_ce': 0.40879, 'rouge_2_recall': 0.23252, 'rouge_2_recall_cb': 0.22532, 'rouge_2_recall_ce': 0.23968, 'rouge_2_precision': 0.14877, 'rouge_2_precision_cb': 0.14415, 'rouge_2_precision_ce': 0.1537, 'rouge_2_f_score': 0.17542, 'rouge_2_f_score_cb': 0.1702, 'rouge_2_f_score_ce': 0.18092, 'rouge_l_recall': 0.48088, 'rouge_l_recall_cb': 0.47359, 'rouge_l_recall_ce': 0.48837, 'rouge_l_precision': 0.31231, 'rouge_l_precision_cb': 0.30728, 'rouge_l_precision_ce': 0.31787, 'rouge_l_f_score': 0.36615, 'rouge_l_f_score_cb': 0.36106, 'rouge_l_f_score_ce': 0.37156}

    p = tempfile.mkdtemp(prefix='/datadrive/tmp/')
    x = test_rouge(p, cand_tok, ref_tok)
    print(x)
    # {'rouge_1_recall': 0.53084, 'rouge_1_recall_cb': 0.52336, 'rouge_1_recall_ce': 0.53812, 'rouge_1_precision': 0.34401, 'rouge_1_precision_cb': 0.33876, 'rouge_1_precision_ce': 0.34965, 'rouge_1_f_score': 0.40365, 'rouge_1_f_score_cb': 0.39861, 'rouge_1_f_score_ce': 0.40887, 'rouge_2_recall': 0.23309, 'rouge_2_recall_cb': 0.22584, 'rouge_2_recall_ce': 0.24023, 'rouge_2_precision': 0.14898, 'rouge_2_precision_cb': 0.14435, 'rouge_2_precision_ce': 0.15392, 'rouge_2_f_score': 0.17574, 'rouge_2_f_score_cb': 0.17048, 'rouge_2_f_score_ce': 0.18132, 'rouge_l_recall': 0.48136, 'rouge_l_recall_cb': 0.47409, 'rouge_l_recall_ce': 0.48881, 'rouge_l_precision': 0.31228, 'rouge_l_precision_cb': 0.30724, 'rouge_l_precision_ce': 0.31784, 'rouge_l_f_score': 0.36628, 'rouge_l_f_score_cb': 0.36119, 'rouge_l_f_score_ce': 0.37174}

    p = tempfile.mkdtemp(prefix='/datadrive/tmp/')
    x = test_rouge(p, cand_detok, ref_detok)
    print(x)
    # {'rouge_1_recall': 0.53095, 'rouge_1_recall_cb': 0.52349, 'rouge_1_recall_ce': 0.53831, 'rouge_1_precision': 0.34405, 'rouge_1_precision_cb': 0.33877, 'rouge_1_precision_ce': 0.34978, 'rouge_1_f_score': 0.40373, 'rouge_1_f_score_cb': 0.39866, 'rouge_1_f_score_ce': 0.40892, 'rouge_2_recall': 0.23301, 'rouge_2_recall_cb': 0.22576, 'rouge_2_recall_ce': 0.24007, 'rouge_2_precision': 0.14891, 'rouge_2_precision_cb': 0.14429, 'rouge_2_precision_ce': 0.15384, 'rouge_2_f_score': 0.17567, 'rouge_2_f_score_cb': 0.17036, 'rouge_2_f_score_ce': 0.18122, 'rouge_l_recall': 0.48148, 'rouge_l_recall_cb': 0.47418, 'rouge_l_recall_ce': 0.48899, 'rouge_l_precision': 0.31231, 'rouge_l_precision_cb': 0.30727, 'rouge_l_precision_ce': 0.31789, 'rouge_l_f_score': 0.36635, 'rouge_l_f_score_cb': 0.36126, 'rouge_l_f_score_ce': 0.37176}
