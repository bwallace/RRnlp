"""
the BiasAbRobot class takes the *abstract* of a clinical trial as
input as a string, and returns bias information as a dict which
can be easily converted to JSON.

V2.0

Returns an indicative probability that the article is at low risk of bias, based on the abstract alone.




"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@northeastern.edu>

import json
import uuid
import os
import rrnlp
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import numpy as np
import re
import scipy
from scipy.sparse import hstack
import rrnlp


weights_path = rrnlp.models.weights_path

doi = rrnlp.models.files_needed['RoB_classifier_LR']['zenodo']

class AbsRoBBot:
    ''' Lightweight container class that holds RoB logistic regression model '''

    def __init__(self):
        
        with open(os.path.join(weights_path, f'{doi}_bias_prob_clf.pck'), 'rb') as f:
            self.clf = pickle.load(f)

        self.vec = HashingVectorizer(ngram_range=(1, 3), stop_words='english')


    def predict_for_ab(self, ab: dict) -> dict: 

        """
        Annotate abstract of clinical trial report
            
        """
        ti_and_abs = ab['ti'] + "  " + ab['ab']
        X = self.vec.transform([ti_and_abs])

        probs = self.clf.predict_proba(X)[:,1].tolist()
        
        return {"prob_low_rob": probs[0]}



###
# e.g.
#
# import RoB_classifier_LR
# RoB_bot = RoB_classifier_LR.AbsRoBBot()
#
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
#
# pred_low_RoB = RoB_bot.predict_for_doc(ti_abs) 
