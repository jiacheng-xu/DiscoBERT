import torch


def read_single_data(d):
    src = d['src']
    segs = d['segs']
    clss = d['clss']
    disco_text = d['disco_txt']
    tgt_txt = d['tgt_txt']
    d_labels = d['d_labels']
    d_span = d['d_span']

    assert len(disco_text) == len(d_labels) == len(d_span)

    print(tgt_txt)
    for x in d['sent_txt']:
        print(" ".join(x))
    # print text

    # for x, y in zip(src, segs):
    #     print("{}\t{}".format(x, y))

    disco_print = []
    for idx, label in enumerate(d_labels):
        if label > 0:
            disco_print.append("|({})[1]".format(idx) + " ".join(disco_text[idx]))
        else:
            disco_print.append("|({})[ ]".format(idx) + " ".join(disco_text[idx]))

    print("COREF")
    for x in d['d_coref']:
        a, b = x
        disco_print[a] = "|{}".format(b) + disco_print[a]

    print("\n".join(disco_print))
    exit()


if __name__ == '__main__':
    file = '/datadrive/data/cnndm/train/dailymail.train.1.bert.pt'
    data = torch.load(file)
    for x in data:
        # print(x)
        read_single_data(x)
