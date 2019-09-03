def percent(inp, start=0.25, end=0.5):
    l = len(inp)
    s = int(l * start)
    e = int(l * end)
    return sum(inp[s:e]) / sum(inp) * 100
    # return sum(inp[:10]) / sum(inp) * 100


if __name__ == '__main__':
    import torch

    x = torch.load('/datadrive/data/dailymail/chunk/dailymail.test.1.bert.pt')
    for every in x:
        # lbs = every['labels'][0]
        lbs = every['d_labels'][0]
        d_coref = every['d_coref']
        disco_txt = every['disco_txt']

        z = percent(lbs, 0.5, 1.0)
        if z > 70:
            # print(every)
            print('-'*20)
            print(" ".join(every['tgt_list_str']))
            print(every['doc_id'])
            did = every['doc_id']
            doc_id = 'b663aefdcb769b1dabcc7cb65e1c089a217c728a.story.doc'
            if did == doc_id:
                ds = every['disco_txt']
                for _d in ds:
                    print(" ".join(_d))
            print(lbs)
            for idx, lb in enumerate(lbs):

                if lb == 1 and idx > 20:
                    sign = False
                    for cor in d_coref:
                        if idx in cor:
                            sign = True
                    if sign:
                        print(idx)
                        print(" ".join(disco_txt[idx])  )