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
  'sensitive': 0.002569688226032838,
  'balanced': 0.001556109585536539},
 'bert_ptyp': {'precise': 0.04210370868268101,
  'sensitive': 0.005688371353031746,
  'balanced': 0.0015407902417415272}}

device = rrnlp.models.device
weights_path = rrnlp.models.weights_path

# These are the paths to the classifier (clf) and (custom; top-k layer)
# encoder weights for the RCT model.
clf_weights_path = os.path.join(weights_path, "RCT_overall_abs_clf.pt")
# Task-specific weights for the encoder
shared_encoder_weights_path = os.path.join(weights_path, "RCT_encoder_custom.pt")

with open(os.path.join(weights_path, "bert_LR.pck"), 'rb') as f:
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
            scores[f"is_rct_{t}"] = (prob_rct > thresholds['bert'][t])[0]


        return {"is_rct": scores['is_rct_balanced'], "prob_rct": (prob_rct)[0], "scores": scores}

    def predict_for_doc(self, ti_and_abs: str) -> float:
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
# title    = '''Repurposed Antiviral Drugs for Covid-19 - Interim WHO Solidarity Trial Results'''
# abstract = '''Background: World Health Organization expert groups recommended mortality trials of four repurposed antiviral drugs - remdesivir, hydroxychloroquine, lopinavir, and interferon beta-1a - in patients hospitalized with coronavirus disease 2019 (Covid-19). Methods: We randomly assigned inpatients with Covid-19 equally between one of the trial drug regimens that was locally available and open control (up to five options, four active and the local standard of care). The intention-to-treat primary analyses examined in-hospital mortality in the four pairwise comparisons of each trial drug and its control (drug available but patient assigned to the same care without that drug). Rate ratios for death were calculated with stratification according to age and status regarding mechanical ventilation at trial entry. Results: At 405 hospitals in 30 countries, 11,330 adults underwent randomization; 2750 were assigned to receive remdesivir, 954 to hydroxychloroquine, 1411 to lopinavir (without interferon), 2063 to interferon (including 651 to interferon plus lopinavir), and 4088 to no trial drug. Adherence was 94 to 96% midway through treatment, with 2 to 6% crossover. In total, 1253 deaths were reported (median day of death, day 8; interquartile range, 4 to 14). The Kaplan-Meier 28-day mortality was 11.8% (39.0% if the patient was already receiving ventilation at randomization and 9.5% otherwise). Death occurred in 301 of 2743 patients receiving remdesivir and in 303 of 2708 receiving its control (rate ratio, 0.95; 95% confidence interval [CI], 0.81 to 1.11; P = 0.50), in 104 of 947 patients receiving hydroxychloroquine and in 84 of 906 receiving its control (rate ratio, 1.19; 95% CI, 0.89 to 1.59; P = 0.23), in 148 of 1399 patients receiving lopinavir and in 146 of 1372 receiving its control (rate ratio, 1.00; 95% CI, 0.79 to 1.25; P = 0.97), and in 243 of 2050 patients receiving interferon and in 216 of 2050 receiving its control (rate ratio, 1.16; 95% CI, 0.96 to 1.39; P = 0.11). No drug definitely reduced mortality, overall or in any subgroup, or reduced initiation of ventilation or hospitalization duration. Conclusions: These remdesivir, hydroxychloroquine, lopinavir, and interferon regimens had little or no effect on hospitalized patients with Covid-19, as indicated by overall mortality, initiation of ventilation, and duration of hospital stay. (Funded by the World Health Organization; ISRCTN Registry number, ISRCTN83971151; ClinicalTrials.gov number, NCT04315948.).'''
# ti_abs   = title + " " + abstract
#
# pred_low_RCT = RCT_bot.predict_for_doc(ti_abs)

