import torch


def extract_n_grams(inp_str, ngram: int = 3, connect_punc='_') -> set:
    inp_list = inp_str.split(" ")
    if len(inp_list) < ngram:
        return set()
    tmp = []
    for idx in range(len(inp_list) - ngram + 1):
        this = [inp_list[idx + j] for j in range(ngram)]
        tmp.append(connect_punc.join(this))
    return set(tmp)


def detect_nan(input_tensor) -> bool:
    if torch.sum(torch.isnan(input_tensor)) > 0:
        return True
    else:
        return False


from typing import List

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}
import re


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def split_sentence_according_to_id(inp, use_disco, disco_map) -> List[List[int]]:
    if len(inp) <= 1:
        return [inp]
    actual_sent_len = disco_map[-1]
    actual_disco_len = len(disco_map)
    if not use_disco:
        return [[x] for x in inp if x < actual_sent_len]
    d = {}
    for x in inp:
        if x >= actual_disco_len:
            continue
        sent = disco_map[x]
        if sent in d:
            d[sent] = d[sent] + [x]
        else:
            d[sent] = [x]
    buff = []
    for k, v in d.items():
        buff.append(v)
    return buff


from typing import List


def easy_post_processing(inp: List):
    if len(inp) < 2:
        return inp

    if inp[-1] in ',;-=':
        inp.pop(-1)
    if inp[0] in ',;-=.!':
        inp.pop(0)
    return inp


def flatten_2d_matrix_to_1d(two_dim_matrix, word_num):
    batch_size, sent_num = two_dim_matrix.shape
    bias = torch.arange(start=0, end=batch_size, dtype=torch.long, device=two_dim_matrix.device,
                        requires_grad=False) * word_num
    bias = bias.view(-1, 1)
    bias = bias.repeat(1, sent_num).view(-1)
    flatten_2d_raw = two_dim_matrix.view(-1)

    return (flatten_2d_raw + bias).long()


def flatten_3d_tensor_to_2d(three_dim_tensor):
    # flatten the first two dim
    shape0, shape1, shape2 = three_dim_tensor.shape
    return three_dim_tensor.view(shape0 * shape1, shape2)


def efficient_head_selection(top_vec, clss):
    assert top_vec.shape[0] == clss.shape[0]
    word_num = top_vec.shape[1]
    batch_size = top_vec.shape[0]
    sent_num = clss.shape[1]
    sent_mask = (clss >= -0.0001).float()  # batch size, max sent num
    # if random.random()<0.01:
    #     print(sent_mask)
    clss_non_neg = torch.nn.functional.relu(clss).long()

    matrix_top_vec = flatten_3d_tensor_to_2d(top_vec)  # batch size, word seq len, hdim
    vec_clss_non_neg = flatten_2d_matrix_to_1d(clss_non_neg, word_num)
    flatten_selected_sent_rep = torch.index_select(matrix_top_vec, 0, vec_clss_non_neg)

    selected_sent_rep = flatten_selected_sent_rep.view(batch_size, sent_num, -1)
    selected_sent_rep = selected_sent_rep * sent_mask.unsqueeze(-1)
    # print(selected_sent_rep.shape)
    # print(sent_mask.shape)
    return selected_sent_rep, sent_mask
