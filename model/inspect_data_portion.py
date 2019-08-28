import torch
import os
import statistics

if __name__ == '__main__':
    path = '/datadrive/data/nyt/test'
    files = os.listdir(path)
    num_of_sent = []
    num_of_EDU = []
    num_of_tok = []
    num_coref_edges = []
    num_rst_edges = []
    num_tok_ref = []
    for f in files:

        x = torch.load(os.path.join(path, f)
                       )
        for single in x:
            disco_txt = single['disco_txt']
            tgt_token = single['tgt_tok_list_list_str']

            d_coref = single['d_coref']
            num_coref_edges.append(len(d_coref) / 2)

            disco_num = len(disco_txt)
            num_of_EDU.append(disco_num)

            toks = sum(disco_txt, [])
            tok_num = len(toks)
            num_of_tok.append(tok_num)

            num_tok_ref.append(len(sum(tgt_token, [])))
            num_rst_edges.append(len(single['d_graph']))
            num_of_sent.append(len(single['sent_txt']))
    print(statistics.mean(num_of_sent))
    print(statistics.mean(num_of_EDU))
    print(statistics.mean(num_of_tok))
    print(statistics.mean(num_rst_edges))
    print(statistics.mean(num_coref_edges))
    print(statistics.mean(num_tok_ref))
