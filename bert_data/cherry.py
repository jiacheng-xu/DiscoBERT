def read_file(fname):
    with open(fname, 'r') as fd:
        liens = fd.read().splitlines()
    lines = [l.replace('<q>', '. ') for l in liens]
    return lines


def read_file_trim(fname):
    with open(fname, 'r') as fd:
        liens = fd.read().splitlines()
    lines = ["<br>".join(l.split('<q>')[:-1]) for l in liens]
    return lines


# x is ref
def comp_rouge(x: str, y: str):
    xs = x.split(" ")
    ys = y.split(" ")
    lx, ly = len(xs), len(ys)
    overlap = set(xs).intersection(set(ys))
    ol = len(overlap)
    rouge_recall_1 = ol / lx
    rouge_pre_1 = ol / ly
    f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
            rouge_recall_1 + rouge_pre_1)
    return f1, rouge_recall_1, rouge_pre_1


if __name__ == '__main__':

    cand = '0.43593_0.2077_0.40553_cand_9_1_22_53_vfnhhuxpcf.txt'
    cand_full = '0.43593_0.2077_0.40553_cand_full_9_1_22_53_vfnhhuxpcf.txt'
    ref = '0.43593_0.2077_0.40553_ref_9_1_22_53_vfnhhuxpcf.txt'

    cand = read_file(cand)
    cand_full = read_file(cand_full)
    ref = read_file(ref)
    l = len(cand)
    tmp = []
    for c, cf, r in zip(cand, cand_full, ref):
        f, prec, recall = comp_rouge(c, r)
        f_full, prec_full, recall_full = comp_rouge(cf, r)
        if recall_full - recall <0.03 and (prec - prec_full >0.2 ):
            print("{:10.4f} {:10.4f} | {:10.4f} {:10.4f} ".format(prec, prec_full, recall, recall_full))
            print("{}\n{}\n{}".format(c, cf,r ))