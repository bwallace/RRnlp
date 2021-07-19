'''
This module extracts descriptions (snippets) that describe the trial
Population, Interventions/Comparators, and Outcomes (PICO elements)
from abstracts of RCT reports.

Reference: 

    Nye, B., Li, J.J., Patel, R., Yang, Y., Marshall, I.J., Nenkova, A. 
        and Wallace, B.C.
    A corpus with multi-level annotations of patients, interventions and 
        outcomes to support language processing for medical literature. 
    In Proceedings of Association for Computational Linguistics (ACL), 2018.
'''


import os
import string 
from typing import Type, Tuple, List

import torch 
from transformers import *

import rrnlp
from rrnlp.models import encoder 
from rrnlp.models.util.minimap import minimap
from rrnlp.models.util.schwartz_hearst import extract_abbreviation_definition_pairs




device = rrnlp.models.device 
weights_path = rrnlp.models.weights_path
doi = rrnlp.models.files_needed['PICO_tagger']['zenodo']

# this dictionary specifies paths to the (torch) weights on disk for
# the P, I, O models (both the classifier or 'clf' layer and the 
# (custom, top layers of the) encoders.
weights_paths = {
    "p" : {"clf": os.path.join(weights_path, f"{doi}_population_clf.pt"),
           "encoder" : os.path.join(weights_path, f"{doi}_population_encoder_custom.pt")},
    "i" : {"clf": os.path.join(weights_path, f"{doi}_interventions_clf.pt"),
           "encoder" : os.path.join(weights_path, f"{doi}_interventions_encoder_custom.pt")}, 
    "o" : {"clf": os.path.join(weights_path, f"{doi}_outcomes_clf.pt"),
           "encoder" : os.path.join(weights_path, f"{doi}_outcomes_encoder_custom.pt")}
}

ids2tags = {
    "p" : {0:'pop', 1:'O'},
    "i" : {0:'intervention', 1:'O'},
    "o" : {0:'outcome', 1:'O'}
}

def get_tagging_model(element: str) -> Type[BertForTokenClassification]:
    ''' Load in and return a tagger for a given element '''

    assert(element in ids2tags.keys())

    # note that we assume the models were trained under I/O
    # encoding such that num_labels is 2
    model = BertForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', 
                                                        num_labels=2)

    
    # load in the correct top layer weights
    clf_weights_path = weights_paths[element]['clf']
    model.classifier.load_state_dict(torch.load(clf_weights_path, 
                                        map_location=torch.device(device)))
    

    encoder_weights_path = weights_paths[element]['encoder']
    custom_encoder_layers = torch.load(encoder_weights_path, 
                                      map_location=torch.device(device))
    encoder.load_encoder_layers(model.bert, encoder.get_muppet(), custom_encoder_layers)
    
    return model 

def print_labels(tokens: List[str], labels: List[str]) -> List[str]:
    ''' Helper to gather strings assigned labels '''
    all_strs, cur_str = [], []
    cur_lbl = "O"
    for token, lbl in zip(tokens, labels):
        if lbl != "O":
            cur_str.append(token)
            cur_lbl = lbl
        elif cur_lbl != "O":
            str_ = " ".join(cur_str)
            all_strs.append(str_)
            cur_str = []
            cur_lbl = "O"
        
    return all_strs

def predict_for_str(model: Type[BertForTokenClassification], string: str, 
                    id2tag: dict, print_tokens: bool=True, o_lbl:str="O", 
                    return_strings_only: bool=True) -> list: 
    ''' 
    Make predictions for the input text using the given tagging model.
    '''
    model.eval()
    words = string.split(" ")
    
    x = encoder.tokenize([words])

    with torch.no_grad():
        preds = model(torch.tensor(x['input_ids']).to(device))['logits'].cpu().numpy().argmax(axis=2)
        preds = [id2tag[p] for p in preds[0]]
        
        cur_w_idx = None
        word_preds = []
        for pred, word_idx in zip(preds, x.word_ids()):
            if word_idx != cur_w_idx and word_idx is not None:
                word_preds.append(pred)
            cur_w_idx = word_idx

        words_and_preds = list(zip(words, word_preds)) 
        if return_strings_only:
            return print_labels(words, word_preds)

        return words_and_preds

def cleanup(spans: List[str]) -> List[str]:
    '''
    A helper (static) function for prettifying / deduplicating the
    PICO snippets extracted by the model.
    '''
    def clean_span(s):
        s_clean = s.strip()
        # remove punctuation
        s_clean = s_clean.strip(string.punctuation)

        # remove 'Background:' when we pick it up
        s_clean = s_clean.replace("Background", "")
        return s_clean


    cleaned_spans = [clean_span(s) for s in spans]
    # remove empty
    cleaned_spans = [s for s in cleaned_spans if s]
    # dedupe
    return list(set(cleaned_spans))


class PICOBot:
    ''' Lightweight class that holds taggers for all elements '''
    def __init__(self):
        self.PICO_models = {}
        for element in ['p', 'i', 'o']: 
            self.PICO_models[element] = get_tagging_model(element)


    def predict_for_ab(self, ab: dict) -> dict:

        ti_abs = ab['ab'].strip()

        preds_d = {}

        
        abbrev_dict = extract_abbreviation_definition_pairs(doc_text=ti_abs)

        for element, model in self.PICO_models.items():
            
            id2tag = ids2tags[element]
            predicted_spans = cleanup(predict_for_str(model, ti_abs, id2tag))
            MeSH_terms = minimap.get_unique_terms(predicted_spans, abbrevs=abbrev_dict)
            preds_d[element] = predicted_spans
            preds_d[f"{element}_mesh"] = MeSH_terms
            
        return preds_d


'''
e.g.,

import PICO_tagger
bot = PICO_tagger.PICOBot()
ti_abs = {"ti": 'A Cluster-Randomized Trial of Hydroxychloroquine for Prevention of Covid-19',
          "ab": """ Background: Current strategies for preventing severe acute
           respiratory syndrome coronavirus 2 (SARS-CoV-2) infection are
           limited to nonpharmacologic interventions. Hydroxychloroquine has
           been proposed as a postexposure therapy to prevent coronavirus
           disease 2019 (Covid-19), but definitive evidence is lacking.

          Methods: We conducted an open-label, cluster-randomized trial
          involving asymptomatic contacts of patients with
          polymerase-chain-reaction (PCR)-confirmed Covid-19 in Catalonia,
          Spain. We randomly assigned clusters of contacts to the
          hydroxychloroquine group (which received the drug at a dose of 800 mg
          once, followed by 400 mg daily for 6 days) or to the usual-care
          group (which received no specific therapy). The primary outcome was
          PCR-confirmed, symptomatic Covid-19 within 14 days. The secondary
          outcome was SARS-CoV-2 infection, defined by symptoms compatible with
          Covid-19 or a positive PCR test regardless of symptoms. Adverse
          events were assessed for up to 28 days.\n\nResults: The analysis
          included 2314 healthy contacts of 672 index case patients with
          Covid-19 who were identified between March 17 and April 28, 2020. A
          total of 1116 contacts were randomly assigned to receive
          hydroxychloroquine and 1198 to receive usual care. Results were
          similar in the hydroxychloroquine and usual-care groups with respect
          to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and
          6.2%, respectively; risk ratio, 0.86 [95% confidence interval, 0.52
          to 1.42]). In addition, hydroxychloroquine was not associated with a
          lower incidence of SARS-CoV-2 transmission than usual care (18.7% and
          17.8%, respectively). The incidence of adverse events was higher in
          the hydroxychloroquine group than in the usual-care group (56.1% vs.
          5.9%), but no treatment-related serious adverse events were
          reported.\n\nConclusions: Postexposure therapy with
          hydroxychloroquine did not prevent SARS-CoV-2 infection or
          symptomatic Covid-19 in healthy persons exposed to a PCR-positive
          case patient. (Funded by the crowdfunding campaign YoMeCorono and
          others; BCN-PEP-CoV2 ClinicalTrials.gov number, NCT04304053.).
          """
}



preds = bot.predict_for_ab(ti_abs)
'''
        