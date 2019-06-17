from typing import Optional

from overrides import overrides
import torch
from typing import List
# from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from pythonrouge.pythonrouge import Pythonrouge
import pickle
import logging
import os, json


@Metric.register("rouge")
class RougeStrEvaluation(Metric):

    def __init__(self, name,
                 path_to_valid: str = '/tmp/',
                 writting_address: str = "",
                 serilization_name: str = "") -> None:
        self.pred_str_bag: List[List[str]] = []
        self.ref_str_bag: List[List[List[str]]] = []
        self.origin_str_bag = []
        self.name_bag: List[str] = []
        self.name = name
        self.writting_address = writting_address
        self.path = path_to_valid
        self.serilization_name = serilization_name

    @overrides
    def __call__(self, pred: List[str], ref: List[List[str]],
                 origin: List[str] = None, **kwargs):
        # # summary: double list
        # summary = [[summaryA_sent1, summaryA_sent2],
        #            [summaryB_sent1, summaryB_sent2]]
        # # reference: triple list
        # reference = [[[summaryA_ref1_sent1, summaryA_ref1_sent2],
        #               [summaryA_ref2_sent1, summaryA_ref2_sent2]],
        #              [[summaryB_ref1_sent1, summaryB_ref1_sent2],
        #               [summaryB_ref2_sent1, summaryB_ref2_sent2]]
        self.pred_str_bag.append(pred)
        self.ref_str_bag.append(ref)
        if origin:
            self.origin_str_bag.append(origin)

    def return_blank_metrics(self):
        all_metrics = {}
        all_metrics[self.name + '_1'] = 0
        all_metrics[self.name + '_2'] = 0
        all_metrics[self.name + '_L'] = 0
        all_metrics[self.name + '_A'] = 0
        self.reset()
        return all_metrics

    def get_metric(self, reset: bool):
        # print("Len of predictions in bag {}".format(len(self.pred_str_bag)))
        if (not reset) or (len(self.pred_str_bag) == 0):
            # print("reset: {}\tVolum: {}\tNote:{}".format(reset,len(self.pred_str_bag),note))
            all_metrics = {}
            all_metrics[self.name + '_1'] = 0
            all_metrics[self.name + '_2'] = 0
            all_metrics[self.name + '_L'] = 0
            all_metrics[self.name + '_A'] = 0
            return all_metrics

        # print(len(self.pred_str_bag))
        logger = logging.getLogger()
        # logger.info("reset: {}\tVolum: {}\tNote:{}".format(reset, len(self.pred_str_bag), note))
        # if 'dm' in self.serilization_name and 'cnn' not in self.serilization_name:
        #     try:
        #         assert abs(len(self.pred_str_bag) - 10397) <= 10
        #     except AssertionError:
        #         print("Len: {}  -- match 10397".format(len(self.pred_str_bag)))
        #         logger.warning("Len: {}  -- match 10397".format(len(self.pred_str_bag)))
        #         return self.return_blank_metrics()
        # if 'dm' not in self.serilization_name and 'cnn' in self.serilization_name:
        #     try:
        #         assert abs(len(self.pred_str_bag) - 1093) <= 5
        #     except AssertionError:
        #         print("Len: {}  -- match 1093".format(len(self.pred_str_bag)))
        #         logger.warning("Len: {}  -- match 1093".format(len(self.pred_str_bag)))
        #         return self.return_blank_metrics()
        #
        # if 'nyt' in self.serilization_name:
        #     try:
        #         assert len(self.pred_str_bag) == 17218
        #     except AssertionError:
        #         logger.warning("Len: {}  -- match 17218".format(len(self.pred_str_bag)))
        #         return self.return_blank_metrics()
        # if 'cnndm' in self.serilization_name:
        #     try:
        #         assert abs(len(self.pred_str_bag) - 11490) <= 10
        #     except AssertionError:
        #         return self.return_blank_metrics()
        # assert note != ""

        rouge = Pythonrouge(summary_file_exist=False,
                            summary=self.pred_str_bag, reference=self.ref_str_bag,
                            n_gram=2, ROUGE_SU4=True, ROUGE_L=True, ROUGE_W=True,
                            ROUGE_W_Weight=1.2,
                            recall_only=False, stemming=True, stopwords=False,
                            word_level=True, length_limit=False, length=50,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=True, samples=1000, favor=True, p=0.5,
                            xml_dir=self.path, peer_path=self.path, model_path=self.path)
        score = rouge.calc_score()
        try:
            print("Name: {}\tScore: {}".format(self.name, score))
        except NameError:
            print("Score: {}".format(score))

        logger.info("Name: {}\tLen: {}\tScore: {}".format(self.name, len(self.pred_str_bag), score))
        all_metrics = {}
        all_metrics[self.name + '_1'] = score['ROUGE-1-F']
        all_metrics[self.name + '_2'] = score['ROUGE-2-F']
        all_metrics[self.name + '_L'] = score['ROUGE-L-F']
        all_metrics[self.name + '_A'] = (score['ROUGE-1-F'] + score['ROUGE-2-F'] + score['ROUGE-L-F']) / 3.
        _ser_name = "{0:.3f},{1:.3f},{2:.3f}-{3}-{4}-{5}".format(score['ROUGE-1-F'], score['ROUGE-2-F'],
                                                                 score['ROUGE-L-F'],
                                                                 self.serilization_name,
                                                                 len(self.pred_str_bag),
                                                                 self.name)
        # f = open(os.path.join(self.writting_address, _ser_name), 'wb')
        # pickle.dump({}, f)
        # f.close()
        # if note == "test":
        #
        #     f = open(os.path.join(self.writting_address, _ser_name), 'wb')
        #     # pickle.dump({}, f)
        #     # f.close()
        #     if 'cnndm' in self.serilization_name:
        #         if score['ROUGE-1-F'] > 0.41:
        #             pickle.dump({"pred": self.pred_str_bag, "ref": self.ref_str_bag, "ori": self.origin_str_bag}, f)
        #         else:
        #             pickle.dump({}, f)
        #     elif 'dm' in self.serilization_name:
        #         if score['ROUGE-1-F'] > 0.418:
        #             pickle.dump({"pred": self.pred_str_bag, "ref": self.ref_str_bag, "ori": self.origin_str_bag}, f)
        #         else:
        #             pickle.dump({}, f)
        #     elif 'cnn' in self.serilization_name:
        #         if score['ROUGE-1-F'] > 0.32:
        #             pickle.dump({"pred": self.pred_str_bag, "ref": self.ref_str_bag, "ori": self.origin_str_bag}, f)
        #         else:
        #             pickle.dump({}, f)
        #     elif 'nyt' in self.serilization_name:
        #         if score['ROUGE-1-F'] > 0.452:
        #             pickle.dump({"pred": self.pred_str_bag, "ref": self.ref_str_bag, "ori": self.origin_str_bag}, f)
        #         else:
        #             pickle.dump({}, f)
        #     else:
        #         raise NotImplementedError("Dataset mismatch!")
        #     f.close()

        if reset:
            self.reset()
        return all_metrics

    @overrides
    def reset(self):
        self.pred_str_bag = []
        self.ref_str_bag = []
        self.origin_str_bag = []
