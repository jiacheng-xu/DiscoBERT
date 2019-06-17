from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('tensor_bert')
class TensorBertSumPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    :class:`~allennlp.models.encoder_decoder.simple_seq2seq` and
    :class:`~allennlp.models.encoder_decoder.copynet_seq2seq`.
    """

    # def predict(self, source: str) -> JsonDict:
    #     return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict,
                          ) -> Instance:
        """
        """
        print(json_dict)
        document = json_dict['doc']
        # assert type(document) is List  # doc: List[ List[str]]
        summary = json_dict['summary']
        ext_oracle_index = json_dict['ext_oracle_index']
        seg_index = json_dict['seg_index']
        seg_index = [x + 1 for x in seg_index]
        #TODO
        raise NotImplementedError
        # run model for two times. One is for global mask, one is for local msk.

        other_data = {}
        return self._dataset_reader.text_to_instance(document,
                                                     summary,
                                                     ext_oracle_index,
                                                     seg_index,
                                                     other_data)
