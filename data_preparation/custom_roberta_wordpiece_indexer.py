# pylint: disable=no-self-use
from typing import Dict, List, Callable
import logging

from overrides import overrides
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)

# TODO(joelgrus): Figure out how to generate token_type_ids out of this token indexer.

from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer


@TokenIndexer.register("roberta-pretrained")
class PretrainedRobertaIndexer(WordpieceIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    truncate_long_sequences : ``bool``, optional (default=``True``)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    """

    def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 never_lowercase: List[str] = [],
                 max_pieces: int = 514 * 2,
                 truncate_long_sequences: bool = True) -> None:
        from transformers import RobertaTokenizer
        roberta_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        from transformers import AutoTokenizer

        super().__init__(vocab=roberta_tokenizer.encoder,
                         wordpiece_tokenizer=roberta_tokenizer._tokenize,
                         namespace="roberta",
                         use_starting_offsets=use_starting_offsets,
                         max_pieces=max_pieces,
                         never_lowercase=never_lowercase,
                         start_tokens=["<s>"],
                         end_tokens=["</s>"],
                         separator_token="</s>",
                         truncate_long_sequences=truncate_long_sequences)
        self._separator_ids = [2]
        self._start_piece_ids = [0]
        self._end_piece_ids = [2]

    def __eq__(self, other):
        if isinstance(other, PretrainedRobertaIndexer):
            for key in self.__dict__:
                if key == 'wordpiece_tokenizer':
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        return {key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key],
                                                             lambda: 1))
                for key, val in tokens.items()}

    @overrides
    def tokens_to_indices(self, tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str):
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # This lowercases tokens if necessary
        text = (token.text.lower()
                if self._do_lowercase and token.text not in self._never_lowercase
                else token.text
                for token in tokens)

        # Obtain a nested sequence of wordpieces, each represented by a list of wordpiece ids
        token_wordpiece_ids = [[self.vocab[wordpiece] for wordpiece in self.wordpiece_tokenizer(token)]
                               for token in text]

        # Flattened list of wordpieces. In the end, the output of the model (e.g., BERT) should
        # have a sequence length equal to the length of this list. However, it will first be split into
        # chunks of length `self.max_pieces` so that they can be fit through the model. After packing
        # and passing through the model, it should be unpacked to represent the wordpieces in this list.
        flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]

        # Similarly, we want to compute the token_type_ids from the flattened wordpiece ids before
        # we do the windowing; otherwise [SEP] tokens would get counted multiple times.
        flat_token_type_ids = _get_token_type_ids(flat_wordpiece_ids, self._separator_ids)

        # The code below will (possibly) pack the wordpiece sequence into multiple sub-sequences by using a sliding
        # window `window_length` that overlaps with previous windows according to the `stride`. Suppose we have
        # the following sentence: "I went to the store to buy some milk". Then a sliding window of length 4 and
        # stride of length 2 will split them up into:

        # "[I went to the] [to the store to] [store to buy some] [buy some milk [PAD]]".

        # This is to ensure that the model has context of as much of the sentence as possible to get accurate
        # embeddings. Finally, the sequences will be padded with any start/end piece ids, e.g.,

        # "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ...".

        # The embedder should then be able to split this token sequence by the window length,
        # pass them through the model, and recombine them.

        # Specify the stride to be half of `self.max_pieces`, minus any additional start/end wordpieces
        window_length = self.max_pieces - len(self._start_piece_ids) - len(self._end_piece_ids)
        stride = window_length // 2

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        offsets = []

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = len(self._start_piece_ids) if self.use_starting_offsets else len(self._start_piece_ids) - 1

        # Count amount of wordpieces accumulated
        pieces_accumulated = 0
        for token in token_wordpiece_ids:
            # Truncate the sequence if specified, which depends on where the offsets are
            next_offset = 1 if self.use_starting_offsets else 0
            if self._truncate_long_sequences and offset + len(token) - 1 >= window_length + next_offset:
                break

            # For initial offsets, the current value of ``offset`` is the start of
            # the current wordpiece, so add it to ``offsets`` and then increment it.
            if self.use_starting_offsets:
                offsets.append(offset)
                offset += len(token)
            # For final offsets, the current value of ``offset`` is the end of
            # the previous wordpiece, so increment it and then add it to ``offsets``.
            else:
                offset += len(token)
                offsets.append(offset)

            pieces_accumulated += len(token)

        if len(flat_wordpiece_ids) <= window_length:
            # If all the wordpieces fit, then we don't need to do anything special
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids)]
            token_type_ids = self._extend(flat_token_type_ids)
        elif self._truncate_long_sequences:
            self._warn_about_truncation(tokens)
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids[:pieces_accumulated])]
            token_type_ids = self._extend(flat_token_type_ids[:pieces_accumulated])
        else:
            # Create a sliding window of wordpieces of length `max_pieces` that advances by `stride` steps and
            # add start/end wordpieces to each window
            # TODO: this currently does not respect word boundaries, so words may be cut in half between windows
            # However, this would increase complexity, as sequences would need to be padded/unpadded in the middle
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids[i:i + window_length])
                                 for i in range(0, len(flat_wordpiece_ids), stride)]

            token_type_windows = [self._extend(flat_token_type_ids[i:i + window_length])
                                  for i in range(0, len(flat_token_type_ids), stride)]

            # Check for overlap in the last window. Throw it away if it is redundant.
            last_window = wordpiece_windows[-1][1:]
            penultimate_window = wordpiece_windows[-2]
            if last_window == penultimate_window[-len(last_window):]:
                wordpiece_windows = wordpiece_windows[:-1]
                token_type_windows = token_type_windows[:-1]

            token_type_ids = [token_type for window in token_type_windows for token_type in window]

        # Flatten the wordpiece windows
        wordpiece_ids = [wordpiece for sequence in wordpiece_windows for wordpiece in sequence]

        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]

        return {index_name: wordpiece_ids,
                f"{index_name}-offsets": offsets,
                f"{index_name}-type-ids": token_type_ids,
                "mask": mask}

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A RoBERTa sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = self._separator_ids
        cls = self._start_piece_ids

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep) * [0] + len(token_ids_1 + sep) * [1]



if __name__ == '__main__':
    x = PretrainedRobertaIndexer('roberta-large')
    # print(x)
    for key, val in x.vocab.items():
        print(key)
