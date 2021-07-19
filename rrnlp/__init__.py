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
                        sample_size_extractor, RoB_classifier_LR, \
                        RCT_classifier

class TrialReader:

    def __init__(self):

        self.models = {"rct_bot": RCT_classifier.AbsRCTBot(),
                       "pico_span_bot": PICO_tagger.PICOBot(),
                       "punchline_bot": ev_inf_classifier.EvInfBot(),
                       "bias_ab_bot": RoB_classifier_LR.AbsRoBBot(),
                       "sample_size_bot": sample_size_extractor.MLPSampleSizeClassifier()}


    def read_trial(self, ab: dict, process_rcts_only=True,
                   task_list=None) -> Type[dict]:
        """
        The default behaviour is that non-RCTs do not have all extractions done (to save time).
        If you wish to use all the models anyway (which might not behave entirely as expected)
        then set `process_rcts_only=False`.
        """

        if task_list is None:
            task_list = ["rct_bot", "pico_span_bot", "punchline_bot",
                   "bias_ab_bot", "sample_size_bot"]

        return_dict = {}

        if process_rcts_only:
            task_list.remove('rct_bot')
            # First: is this an RCT? If not, the rest of the models do not make
            # a lot of sense so we will warn the user
            return_dict["rct_bot"] = self.models['rct_bot'].predict_for_ab(ab)

        if not return_dict["rct_bot"]["is_rct"]:
            if process_rcts_only:
                print('''Predicted as non-RCT, so rest of models not run. Re-run
                         with `process_rcts_only=False` to get all predictions.''')
            else:
                print('''Warning! The input does not appear to describe an RCT; 
                         interpret predictions accordingly.''')

        if (not process_rcts_only) or return_dict["rct_bot"]["is_rct"]:

            for task in task_list:
                return_dict[task] = self.models[task].predict_for_ab(ab)
                                            
        return return_dict


# For e.g.:
# import rrnlp  
# trial_reader = rrnlp.TrialReader()
    
# ti_abs = {"ti": 'A Cluster-Randomized Trial of Hydroxychloroquine for Prevention of Covid-19',
#           "ab": """ Background: Current strategies for preventing severe acute
#            respiratory syndrome coronavirus 2 (SARS-CoV-2) infection are
#            limited to nonpharmacologic interventions. Hydroxychloroquine has
#            been proposed as a postexposure therapy to prevent coronavirus
#            disease 2019 (Covid-19), but definitive evidence is lacking.

#           Methods: We conducted an open-label, cluster-randomized trial
#           involving asymptomatic contacts of patients with
#           polymerase-chain-reaction (PCR)-confirmed Covid-19 in Catalonia,
#           Spain. We randomly assigned clusters of contacts to the
#           hydroxychloroquine group (which received the drug at a dose of 800 mg
#           once, followed by 400 mg daily for 6 days) or to the usual-care
#           group (which received no specific therapy). The primary outcome was
#           PCR-confirmed, symptomatic Covid-19 within 14 days. The secondary
#           outcome was SARS-CoV-2 infection, defined by symptoms compatible with
#           Covid-19 or a positive PCR test regardless of symptoms. Adverse
#           events were assessed for up to 28 days.\n\nResults: The analysis
#           included 2314 healthy contacts of 672 index case patients with
#           Covid-19 who were identified between March 17 and April 28, 2020. A
#           total of 1116 contacts were randomly assigned to receive
#           hydroxychloroquine and 1198 to receive usual care. Results were
#           similar in the hydroxychloroquine and usual-care groups with respect
#           to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and
#           6.2%, respectively; risk ratio, 0.86 [95% confidence interval, 0.52
#           to 1.42]). In addition, hydroxychloroquine was not associated with a
#           lower incidence of SARS-CoV-2 transmission than usual care (18.7% and
#           17.8%, respectively). The incidence of adverse events was higher in
#           the hydroxychloroquine group than in the usual-care group (56.1% vs.
#           5.9%), but no treatment-related serious adverse events were
#           reported.\n\nConclusions: Postexposure therapy with
#           hydroxychloroquine did not prevent SARS-CoV-2 infection or
#           symptomatic Covid-19 in healthy persons exposed to a PCR-positive
#           case patient. (Funded by the crowdfunding campaign YoMeCorono and
#           others; BCN-PEP-CoV2 ClinicalTrials.gov number, NCT04304053.).
#           """
# }
# preds = trial_reader.read_trial(ti_abs)