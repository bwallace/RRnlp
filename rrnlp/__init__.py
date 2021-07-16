'''
A module that ties together the constituent models into a single
interface; will pull everything it can from an input article.

Note: if you find this useful, please see: 
        https://github.com/bwallace/RRnlp#citation.
'''
from typing import Type, Tuple, List

import warnings

import rrnlp
from rrnlp.models import PICO_tagger, ev_inf_classifier, \
                        sample_size_extractor, RoB_classifier, \
                        RCT_classifier

class TrialReader:

    def __init__(self):   
        self.RCT_model = RCT_classifier.AbsRCTBot()
        self.pico_model = PICO_tagger.PICOBot()
        self.inf_model  = ev_inf_classifier.EvInfBot()
        self.ss_model   = sample_size_extractor.MLPSampleSizeClassifier()
        self.rob_model  = RoB_classifier.AbsRoBBot()

    def read_trial(self, abstract_text: str, process_rcts_only=True) -> Type[dict]:
        """
        The default behaviour is that non-RCTs do not have all extractions done (to save time).
        If you wish to use all the models anyway (which might not behave entirely as expected)
        then set `process_rcts_only=False`.
        """
        return_dict = {}

        # First: is this an RCT? If not, the rest of the models do not make
        # a lot of sense so we will warn the user
        return_dict["rct_bot"] = self.RCT_model.predict_for_doc(abstract_text)

        if not return_dict["rct_bot"]["is_rct"]:
            if process_rcts_only:
                print('''Predicted as non-RCT, so rest of models not run. Re-run
                         with `process_rcts_only=False` to get all predictions.''')
            else:
                print('''Warning! The input does not appear to describe an RCT; 
                         interpret predictions accordingly.''')

        if (not process_rcts_only) or return_dict["rct_bot"]["is_rct"]:
                                            
                return_dict["pico_span_bot"] = self.pico_model.make_preds_for_abstract(abstract_text)
                return_dict["punchline_bot"] = self.inf_model.infer_evidence(abstract_text)
                return_dict["sample_size_bot"] = self.ss_model.predict_for_abstract(abstract_text)
                return_dict["bias_ab_bot"] = self.rob_model.predict_for_doc(abstract_text)
        return return_dict


