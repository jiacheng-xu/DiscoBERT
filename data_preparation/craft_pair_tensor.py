import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(10,10))
np.set_printoptions(precision=2)
import seaborn as sns

sns.set()

if __name__ == '__main__':
    path = '/datadrive/data/cnndm/train'
    os.chdir(path)
    from data_preparation.search_algo import appx_simple_rouge_estimator

    file = 'dailymail.train.141.bert.pt'
    dataset = torch.load(os.path.join(file))
    for d in dataset:
        print(d)
        sent = d['sent_txt'][:20]

        tgt = d['tgt_tok_list_list_str']

        uni_f_score_list = [appx_simple_rouge_estimator(
            sent[i], tgt
        ) for i in range(len(sent))]
        uni_f_score_list = [int(x * 100) for x in uni_f_score_list]
        uni_f_score = np.asarray(uni_f_score_list).reshape((-1, 1))
        # plt.figure(figsize=(3, 16))
        from data_preparation.nlpyang_utils import _get_word_ngrams, _get_ngrams

        # 2d
        bin_f_score_list = [[0 for _ in range(len(sent))] for _ in range(len(sent))]
        bin_f_label_list = [[0 for _ in range(len(sent))] for _ in range(len(sent))]
        for i in range(len(sent)):
            for j in range(len(sent)):
                pair = int(appx_simple_rouge_estimator(sent[i] + sent[j], tgt) * 100)
                bin_f_score_list[i][j] = pair

                if pair > max(uni_f_score_list[i], uni_f_score_list[j]) * 0.7:
                    bin_f_label_list[i][j] = 1
                else:
                    bin_f_label_list[i][j] = 0

                # itri = _get_word_ngrams(2, [sent[i]])
                # jtri = _get_word_ngrams(2, [sent[j]])
                # if itri.isdisjoint(jtri):
                #     bin_f_label_list[i][j] = 1
                # else:
                #     bin_f_label_list[i][j] = 0

        bin_f_score = np.asarray(bin_f_score_list)

        # f_score = uni_f_score
        # plt.figure(figsize=(3, 16))

        # f_score = bin_f_score
        # plt.figure(figsize=(16, 16))

        f_score = bin_f_label_list
        plt.figure(figsize=(16, 16))

        bx = sns.heatmap(f_score, vmin=0, vmax=1,
                         square=True,
                         linewidths=1,
                         annot=True, fmt="d")
        fig = bx.get_figure()

        fig.savefig("individual_output.png")

        exit()
    f_score_list = np.asarray([f_score_list, f_score_list])
    bx = sns.heatmap(f_score_list)
    fig = bx.get_figure()
    fig.savefig("individual_output.png")
    print('-' * 30)
    print(np.asarray(score_matrix))
    score_matrix_delta = np.asarray(score_matrix_delta)
    ax = sns.heatmap(score_matrix_delta)
    fig = ax.get_figure()
    fig.savefig("output.png")
