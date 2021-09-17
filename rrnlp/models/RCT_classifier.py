'''
This module classifies input abstracts as describing a Randomized Controlled Trial
(in humans) or not. 

For reference (and citation), see:

    Marshall, Iain J., Anna Noel‐Storr, Joël Kuiper, James Thomas, and Byron C. Wallace. 
    "Machine learning for identifying randomized controlled trials: an evaluation and 
    practitioner's guide." Research synthesis methods 9, no. 4 (2018): 602-614.
'''

import os
from typing import Type, Tuple, List

import torch
from transformers import *

import rrnlp
from rrnlp.models import encoder


import pickle



# Thresholds evaluated via bootstrap on Clinical hedges
thresholds = {'bert': {'precise': 0.007859864302367195,
  'sensitive': 0.0027666038410490913,
  'balanced': 0.005165116927458068},
 'bert_ptyp': {'precise': 0.04210370868268101,
  'sensitive': 0.040919136397870086,
  'balanced': 0.0034764010192827734}}

device = rrnlp.models.device
weights_path = rrnlp.models.weights_path
doi = rrnlp.models.files_needed['RCT_classifier']['zenodo']

# These are the paths to the classifier (clf) and (custom; top-k layer)
# encoder weights for the RCT model.
clf_weights_path = os.path.join(weights_path, f"{doi}_RCT_overall_abs_clf.pt")
# Task-specific weights for the encoder
shared_encoder_weights_path = os.path.join(weights_path, f"{doi}_RCT_encoder_custom.pt")

with open(os.path.join(weights_path, f"{doi}_bert_LR.pck"), 'rb') as f:
    lr = pickle.load(f)

def get_RCT_model() -> Type[BertForSequenceClassification]:
    ''' Load in and return RCT model weights. '''

    # Note that we assume the models were trained under I/O encoding
    # such that num_labels is 2
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased',
                                                        num_labels=2)


    # Read in encoder: a mix of shared weights and custom
    custom_encoder_layers = torch.load(shared_encoder_weights_path,
                                        map_location=torch.device(device))

    encoder.load_encoder_layers(model.bert, encoder.get_muppet(), custom_encoder_layers)

    # Load in the correct top layer (classifier) weights
    model.classifier.load_state_dict(torch.load(clf_weights_path,
                                        map_location=torch.device(device)))
    model.to(device)
    return model


class AbsRCTBot:
    ''' Lightweight container class that holds RCT model '''
    def __init__(self):
        self.RCT_model = get_RCT_model()
        self.RCT_model.eval()

    def classify(self, raw_bert_score: float) -> dict:
        """
        gets balanced classification, but also returns full scores
        """
        prob_rct = lr.predict_proba([[raw_bert_score]])[:,1]

        scores = {}
        for t in ['sensitive', 'balanced', 'precise']:
            scores[f"is_rct_{t}"] = bool((prob_rct > thresholds['bert'][t])[0])


        return {"is_rct": scores['is_rct_balanced'], "prob_rct": float((prob_rct)[0]), "scores": scores}

    def predict_for_ab(self, ab: dict) -> float:
        ti_and_abs = ab['ti'] + '  ' + ab['ab']
        ''' Predicts p(low risk of bias) for input abstract '''
        x = encoder.tokenize(ti_and_abs, is_split_into_words=False)

        with torch.no_grad():

            x_input_ids = torch.tensor(x['input_ids']).to(device).unsqueeze(dim=0)
            attention_mask= torch.tensor(x['attention_mask']).to(device).unsqueeze(dim=0)

            logits = self.RCT_model(x_input_ids, attention_mask=attention_mask)['logits'].cpu()
            probs  = torch.nn.functional.softmax(logits, dim=1).numpy()

            raw_rct_score = probs[0][1]
            return self.classify(raw_rct_score)

    def make_preds_for_abstract(self, ti_and_abs: str) -> float:
        self.predict_for_doc(ti_and_abs)


###
# e.g.
#
# import RCT_classifier
# RCT_bot = RCT_classifier.AbsRCTBot()
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
# pred_RCT = RCT_bot.predict_for_doc(ti_abs)

