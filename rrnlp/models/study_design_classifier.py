'''
This module predicts the most likely study design for a given article.
This version uses a series of simple logistic regression classifier.
'''

import os
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import rrnlp


weights_path = rrnlp.models.weights_path
doi = rrnlp.models.files_needed['study_design_classifier']['zenodo']



vec = HashingVectorizer(ngram_range=(1, 4), stop_words='english')




study_designs = {"sr": "Systematic review",
                 'cohort': "Cohort study",
                 "consensus": "Consensus statement",
                 "ct": "Clinical trial (non-randomized)",
                 "ct_protocol": "Clinical trial protocol",
                 "guideline": "Clinical guideline",
                 "qual": "Qualitative study",
                 "rct": "Randomized controlled trial"}


study_design_clfs = {}


def get_models():
    ''' Load in and return RCT model weights. '''

    models = {}

    for sd in study_designs.keys():
        with open(os.path.join(weights_path, f"{doi}_{sd}_lr.pck"), 'rb') as f:
            models[sd] = pickle.load(f)

    return models


class AbsStudyDesignBot:
    ''' Lightweight container class that holds study design model '''
    def __init__(self):
        self.models = get_models()        

    def predict_for_ab(self, ab: dict) -> float:
        ti_and_abs = ab['ti'] + '  ' + ab['ab']
        ''' Predicts p(low risk of bias) for input abstract '''

        probs = []

        x = vec.transform([ti_and_abs])
        for sd, clf in self.models.items():
            pred = clf.predict_proba(x)[:,1]
            probs.append((sd, float(pred[0])))

        out = {}

        most_likely = max(probs, key=lambda x: x[1])        
        if most_likely[1] >= 0.5:
            out['study_design'] = most_likely[0]
        else:
            out['study_design'] = 'unknown'

        for sd, pred in probs:
            out[f"prob_{sd}"] = pred 
            out[f"is_{sd}"] = bool(pred >=0.5)


        return out


###
# e.g.
#
# from rrnlp import study_design_classifier
# sd_bot = study_design_classifier.AbsStudyDesignBot()

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
# pred_sd = sd_bot.predict_for_ab(ti_abs)


