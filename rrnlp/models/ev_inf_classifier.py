'''
This module performs the "evidence inference" task, using a simple
"pipelined" approach in which we first try and identify a "punchline"
sentence, and then infer the directionality of the evidence that 
seems to be reported in this. 

References: 

    Inferring Which Medical Treatments Work from Reports of Clinical Trials. 
    Eric Lehman, Jay DeYoung, Regina Barzilay, and Byron C. Wallace. 
    Proceedings of the North American Chapter of the Association for Computational 
    Linguistics (NAACL), 2019.

    Evidence Inference 2.0: More Data, Better Models. Jay DeYoung, Eric Lehman, 
        Iain J. Marshall, and Byron C. Wallace. 
    Proceedings of BioNLP (co-located with ACL), 2020.
'''

import os
import sys 
from typing import Type, Tuple, List

import numpy as np 

import torch 
from transformers import *

import rrnlp
from rrnlp.models import encoder 

device = rrnlp.models.device 
weights_path = rrnlp.models.weights_path

doi = rrnlp.models.files_needed['ev_inf_classifier']['zenodo']

# Paths to model weights for both the "punchline" extractor model and the 
# "inference" model. Both comprise custom encoder layers and a top layer
# weight vector.
clf_punchline_weights_path        = os.path.join(weights_path, f"{doi}_evidence_identification_clf.pt") 
shared_enc_punchline_weights_path = os.path.join(weights_path, f"{doi}_evidence_identification_encoder_custom.pt") 

clf_inference_weights_path = os.path.join(weights_path, f"{doi}_inference_clf.pt")
shared_enc_inference_weights_path = os.path.join(weights_path, f"{doi}_inference_encoder_custom.pt")

def get_punchline_extractor() -> Type[BertForSequenceClassification]:
    ''' 
    Returns the 'punchline' extractor, which seeks out sentences that seem to convey
    main findings. 
    '''
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', 
                                                        num_labels=2)

    # Overwrite some of the encoder layers with custom weights.
    custom_encoder_layers = torch.load(shared_enc_punchline_weights_path, map_location=torch.device(device))
    encoder.load_encoder_layers(model.bert, encoder.get_muppet(), custom_encoder_layers)

    # Load in the correct top layer weights.
    model.classifier.load_state_dict(torch.load(clf_punchline_weights_path, 
                                        map_location=torch.device(device)))
  
    return model 


def get_inference_model() -> Type[BertForSequenceClassification]:
    '''
    This is a three-way classification model that attempts to classify punchline
    sentences as reporting a result where the intervention resulted in a sig. 
    decrease, no diff, or sig. increase w/r/t the outcome measured.
    '''
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', 
                                                        num_labels=3)

    # Overwrite some of the encoder layers with custom weights
    custom_encoder_layers = torch.load(shared_enc_inference_weights_path, 
                                        map_location=torch.device(device))
    encoder.load_encoder_layers(model.bert, encoder.get_muppet(), custom_encoder_layers)

    # Load in the correct top layer weights
    model.classifier.load_state_dict(torch.load(clf_inference_weights_path, 
                                        map_location=torch.device(device)))
    return model 


class PunchlineExtractorBot:
    ''' Lightweight container class for extracting punchlines. '''

    def __init__(self):
        self.punchline_extractor_model = get_punchline_extractor()
        self.punchline_extractor_model.eval()

    def predict_for_sentences(self, sents: List[str]) -> Type[np.array]: 
        
        x = encoder.tokenize(sents, is_split_into_words=False)

        with torch.no_grad():
            
            x_input_ids = torch.tensor(x['input_ids']).to(device)
            attention_mask= torch.tensor(x['attention_mask']).to(device)

            logits = self.punchline_extractor_model(x_input_ids, attention_mask=attention_mask)['logits'].cpu()
            probs  = torch.nn.functional.softmax(logits, dim=1).numpy()

            return probs 

    def predict_for_ab(self, ab: dict) -> Tuple[str, float]:
        ti_and_abs = ab['ti'] + '  ' + ab['ab']
        # Split into sentences via scispacy
        sentences = [s.text for s in encoder.nlp(ti_and_abs).sents]
        # Make punchline predictions
        pred_probs = self.predict_for_sentences(sentences)
        best_sent_idx = np.argmax(pred_probs[:,1])
        # Retrieve highest scoring sentence
        best_sent = sentences[best_sent_idx]
        return best_sent, pred_probs[best_sent_idx][1]


class InferenceBot:
    ''' Container for *inference* model which classifies punchlines. '''
    def __init__(self):
        self.inference_model = get_inference_model()
        self.inference_model.eval()

    def predict_for_sentence(self, sent: str) -> Type[np.array]: 
        ''' 
        Make a threeway pred for the given sentence: Is this punchline
        reporting a sig. decrease (-1), no diff (0), or sig increase (1)? 
        '''
        if type(sent) == str: 
            sent = [sent]

        x = encoder.tokenize(sent, is_split_into_words=False)

        with torch.no_grad():
            
            x_input_ids = torch.tensor(x['input_ids']).to(device)
            attention_mask= torch.tensor(x['attention_mask']).to(device)

            logits = self.inference_model(x_input_ids, attention_mask=attention_mask)['logits'].cpu()
            probs  = torch.nn.functional.softmax(logits, dim=1).numpy()

            return probs 

class EvInfBot:
    ''' Composes the punchline extractor and inference model. '''
    def __init__(self):
        self.pl_bot  = PunchlineExtractorBot()
        self.inf_bot = InferenceBot()

        self.direction_strs = ["↓ sig. decrease", "— no diff", "↑ sig. increase"]

    def predict_for_ab(self, ab: dict) -> Tuple[str, str]:

        
        # Get punchline.
        pl_sent, pred_probs = self.pl_bot.predict_for_ab(ab)

        # Infer direction.
        direction_probs = self.inf_bot.predict_for_sentence(pl_sent) 
        return {"punchline_text": pl_sent, "effect": self.direction_strs[np.argmax(direction_probs)]}

###
# e.g.
# 
# from rrnlp.models import ev_inf_classifier
# pl_bot = ev_inf_classifier.PunchlineExtractorBot()
#
# sentence = ["patients in group b died more often"]
# pred_punchline = pl_bot.predict_for_sentences(sentence) 
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
# sent, prob = pl_bot.predict_for_ab(ti_abs)
#
# inf_bot = ev_inf_classifier.InferenceBot()
# inf_bot.predict_for_sentence(sent)
# 
# OR in one swoop...
#
# ev_bot = ev_inf_classifier.EvInfBot()
# ev_bot.predict_for_ab(ti_abs)
        
