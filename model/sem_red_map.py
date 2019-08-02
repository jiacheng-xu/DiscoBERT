#
"""
Semantic Salience & Redundancy (SSR) Map


Salience Map: emphasize on R[A+B]   (marginalization could reduce to R[X])

Redundancy Map: emphasize on R[A+B] / max(R[A], R[B])


Core algorithm:
Readout: Works similarly as trigram blocking. We still sort the score according to the uni vector

getattr(self, f"self_attention_{i}")

TODO Efficient score computing
TODO Set up key names
"""

"""
Given a document, MapKiosk could provide all of the semantic maps you need.
"""
class MapKiosk():

    def __init__(self):
        pass

    def single_entry_entrance(self):
        pass

    @staticmethod
    def dedup_cal_rouge(evaluated_ngrams: set, reference_ngrams: set, evaluated_len: int, reference_len: int):
        # reference_count = len(reference_ngrams)
        # evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_len == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_len

        if reference_len == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_len

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return {"f": f1_score, "p": precision, "r": recall}

    def get_salience_map(self):
        pass

    def get_redundancy_map(self):
        pass

    def get_ngram_map(self):
        pass

    @staticmethod
    def get_ngrams():
        pass
