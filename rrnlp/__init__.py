import warnings

import rrnlp
from rrnlp.models import PICO_tagger, ev_inf_classifier, sample_size_extractor, RoB_classifier

class TrialReader:

    def __init__(self):   
        self.pico_model = PICO_tagger.PICOBot()
        self.inf_model  = ev_inf_classifier.EvInfBot()
        self.ss_model   = sample_size_extractor.MLPSampleSizeClassifier()
        self.rob_model  = RoB_classifier.AbsRoBBot()

    def read_trial(self, abstract_text):
        return_dict = {}
        return_dict["PICO"]   = self.pico_model.make_preds_for_abstract(abstract_text)
        return_dict["ev_inf"] = self.inf_model.infer_evidence(abstract_text)
        return_dict["n"]      = self.ss_model.predict_for_abstract(abstract_text)
        return_dict["p_low_RoB"]    = self.rob_model.predict_for_doc(abstract_text)
        return return_dict


