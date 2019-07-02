from data_preparation.search_algo import greedy_selection
from pytorch_pretrained_bert import BertTokenizer


class MSBertData():
    def __init__(self, min_src_ntokens, max_src_ntokens, min_nsents, max_nsents):
        self.min_src_ntokens = min_src_ntokens
        self.max_src_ntokens = max_src_ntokens
        self.min_nsents = min_nsents
        self.max_nsents = max_nsents
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess_sent(self, sent_bag, summary, oracle_size=3):
        # TO have: src_subtoken_idxs [for bert encoder], labels[sent level],
        # segments_ids[for bert encoder],
        # cls_ids[for sent level],
        # entity coref linking edge [ sent level and discourse level]
        # src_txt, tgt_txt
        docs = [s.raw_words for s in sent_bag]
        original_src_txts = []
        oracle_ids = greedy_selection(docs, summary, oracle_size)
        # oracle
        labels = [0] * len(sent_bag)
        for l in oracle_ids:
            labels[l] = 1

        src_tokens = []
        for idx, s in enumerate(sent_bag):
            bpes = s.get_bpe_w_cls_sep()
            src_tokens += bpes
            l = s.get_length_w_pad()
            original_src_txts.append(s.raw_words)
        src_tok_index = self.tokenizer.convert_tokens_to_ids(src_tokens)
        _segs = [-1] + [i for i, t in enumerate(src_tok_index) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_tok_index) if t == self.cls_vid]
        tgt_txt = '<q>'.join([' '.join(tt) for tt in summary])
        return src_tok_index, labels, segments_ids, cls_ids, original_src_txts, tgt_txt

    def preprocess_disc(self, disco_bag, summary, oracle_size):
        # oracle labels
        docs = [disc.get_readable_words_as_list() for disc in disco_bag]

        oracle_ids = greedy_selection(docs, summary, oracle_size)
        labels = [0] * len(disco_bag)
        for l in oracle_ids:
            labels[l] = 1

        # coref biiiigggg matrix
        coref_lookup_dict = {}
        # coref = [[False for _ in range(len(disco_bag))] for _ in range(len(disco_bag))]
        coref_list = []
        # global span
        spans = []
        last_sent_idx = -1
        accumulated_length = -1
        for idx, disc in enumerate(disco_bag):
            if disc.sent_idx != last_sent_idx:
                accumulated_length += 2
                last_sent_idx = disc.sent_idx
            _start, _end = accumulated_length, accumulated_length + len(disc.bert_word_pieces)

            disco_bag[idx].glob_start_idx = _start
            disco_bag[idx].glob_end_idx = _end
            spans.append((_start, _end))
            accumulated_length += len(disc.bert_word_pieces)

            for m in disc.mentions:
                coref_lookup_dict[m] = idx
        for idx, disc in enumerate(disco_bag):
            crf = disc.corefs
            for c in crf:
                if c in coref_lookup_dict:
                    coref_list.append((idx, coref_lookup_dict[c]))
        coref_set = set(coref_list)
        # labels[discourse level],
        # span indexs [ for discourse level]
        # entity coref linking edge [discourse level]
        # skip for now: discourse linking edge [discourse level only]
        return labels, spans, coref_set
