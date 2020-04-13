import csv


def csv_wt(fname, rows):
    with open(fname, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['p1', 'p2', 'p3', 'l1', 'l2', 'l3'])
        for r in rows:
            spamwriter.writerow(r)


def read_file(fname):
    with open(fname, 'r') as fd:
        liens = fd.read().splitlines()
    lines = [l.replace('<q>', '<br>') for l in liens]
    return lines


def read_file_trim(fname):
    with open(fname, 'r') as fd:
        liens = fd.read().splitlines()
    lines = ["<br>".join(l.split('<q>')[:-1]) for l in liens]
    return lines


if __name__ == '__main__':
    fname = 'discobert_human_eval.csv'

    cand = '0.43593_0.2077_0.40553_cand_9_1_22_53_vfnhhuxpcf.txt'
    cand_full = '0.43593_0.2077_0.40553_cand_full_9_1_22_53_vfnhhuxpcf.txt'
    ref = '0.43593_0.2077_0.40553_ref_9_1_22_53_vfnhhuxpcf.txt'

    cand = read_file(cand)
    cand_full = read_file_trim(cand_full)
    ref = read_file(ref)
    l = len(cand)
    import random

    tup = [[cand[x], cand_full[x], ref[x]] for x in range(l)]
    random.shuffle(tup)
    tup = tup[:100]
    rows = []
    for t in tup:
        c, cf, r = t

        # c
        cs = c.split('<br>')
        cs = [_c for _c in cs if len(_c) > 25]
        c = "<br>".join(cs)
        budget = len(c)

        # cf
        cfs = cf.split("<br>")
        _bag = ""
        i = 0
        while len(_bag) < budget:
            if i >= len(cfs):
                break
            _bag += cfs[i] + '<br>'
            i += 1
        t = c, _bag, r
        names = ['edu', 'sent', 'ref']
        x = [0, 1, 2]
        random.shuffle(x)
        tmp = []
        tmp.append(t[x[0]])
        tmp.append(t[x[1]])
        tmp.append(t[x[2]])
        tmp.append(names[x[0]])
        tmp.append(names[x[1]])
        tmp.append(names[x[2]])
        rows.append(tmp)
    # print(rows)
    csv_wt(fname, rows)
