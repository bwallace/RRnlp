import os
import sys 

import numpy as np 

import torch 
from transformers import *

import rrnlp
from rrnlp.models import encoder 

device = rrnlp.models.device 
weights_path = rrnlp.models.weights_path

clf_punchline_weights_path        = os.path.join(weights_path, "evidence_identification_clf.pt") 
shared_enc_punchline_weights_path = os.path.join(weights_path, "evidence_identification_encoder_shared.pt") 

clf_inference_weights_path = os.path.join(weights_path, "inference_clf.pt")
shared_enc_inference_weights_path = os.path.join(weights_path, "inference_encoder_shared.pt")

def get_punchline_extractor():
    # note that we assume the models were trained under I/O
    # encoding such that num_labels is 2
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', 
                                                        num_labels=2)

    # overwrite some of the encoder layers with custom weights
    custom_encoder_layers = torch.load(shared_enc_punchline_weights_path, map_location=torch.device(device))
    encoder.load_encoder_layers(model.bert, encoder.get_muppet(), custom_encoder_layers)

    # load in the correct top layer weights
    model.classifier.load_state_dict(torch.load(clf_punchline_weights_path, 
                                        map_location=torch.device(device)))
  
    return model 



def get_inference_model():
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', 
                                                        num_labels=3)

    model.bert = encoder.get_muppet()

    # load in the correct top layer weights
    model.classifier.load_state_dict(torch.load(clf_inference_weights_path, 
                                        map_location=torch.device(device)))
    return model 


class PunchlineExtractorBot:
    def __init__(self):
        self.punchline_extractor_model = get_punchline_extractor()
        self.punchline_extractor_model.eval()

    def predict_for_sentences(self, sents): 
        
        x = encoder.tokenize(sents, is_split_into_words=False)

        with torch.no_grad():
            
            x_input_ids = torch.tensor(x['input_ids']).to(device)
            attention_mask= torch.tensor(x['attention_mask']).to(device)

            logits = self.punchline_extractor_model(x_input_ids, attention_mask=attention_mask)['logits'].cpu()
            probs  = torch.nn.functional.softmax(logits, dim=1).numpy()

            return probs 

    def make_preds_for_abstract(self, ti_and_abs):
        # split into sentences via scispacy
        sentences = [s.text for s in encoder.nlp(ti_and_abs).sents]
        pred_probs = self.predict_for_sentences(sentences)
        best_sent_idx = np.argmax(pred_probs[:,1])
        best_sent = sentences[best_sent_idx]
        return best_sent, pred_probs[best_sent_idx][1]


class InferenceBot:
    def __init__(self):
        self.inference_model = get_inference_model()
        self.inference_model.eval()

    def predict_for_sentence(self, sent): 
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
    ''' Compose the punchline extractor and inference model '''
    
    def __init__(self):
        self.pl_bot  = PunchlineExtractorBot()
        self.inf_bot = InferenceBot()

        self.direction_strs = ["↓ sig. decrease", "— no diff", "↑ sig. increase"]

    def infer_evidence(self, ti_and_abs):
        # get punchline
        pl_sent, pred_probs = self.pl_bot.make_preds_for_abstract(ti_and_abs)

        # infer direction
        direction = self.inf_bot.predict_for_sentence(pl_sent) 

        return pl_sent, self.direction_strs[np.argmax(pred_probs)]

###
# e.g.
#
# import ev_inf_classifier
# pl_bot = ev_inf_classifier.PunchlineExtractorBot()
#
# sentence = ["patients in group b died more often"]
# pred_punchline = pl_bot.predict_for_sentences(sentence) 
#
# abstract = '''Background: The FIDELIO-DKD trial (Finerenone in Reducing Kidney Failure and Disease Progression in Diabetic Kidney Disease) evaluated the effect of the nonsteroidal, selective mineralocorticoid receptor antagonist finerenone on kidney and cardiovascular outcomes in patients with chronic kidney disease and type 2 diabetes with optimized renin-angiotensin system blockade. Compared with placebo, finerenone reduced the composite kidney and cardiovascular outcomes. We report the effect of finerenone on individual cardiovascular outcomes and in patients with and without history of atherosclerotic cardiovascular disease (CVD). Methods: This randomized, double-blind, placebo-controlled trial included patients with type 2 diabetes and urine albumin-to-creatinine ratio 30 to 5000 mg/g and an estimated glomerular filtration rate ≥25 to <75 mL per min per 1.73 m2, treated with optimized renin-angiotensin system blockade. Patients with a history of heart failure with reduced ejection fraction were excluded. Patients were randomized 1:1 to receive finerenone or placebo. The composite cardiovascular outcome included time to cardiovascular death, myocardial infarction, stroke, or hospitalization for heart failure. Prespecified cardiovascular analyses included analyses of the components of this composite and outcomes according to CVD history at baseline. Results: Between September 2015 and June 2018, 13 911 patients were screened and 5674 were randomized; 45.9% of patients had CVD at baseline. Over a median follow-up of 2.6 years (interquartile range, 2.0-3.4 years), finerenone reduced the risk of the composite cardiovascular outcome compared with placebo (hazard ratio, 0.86 [95% CI, 0.75-0.99]; P=0.034), with no significant interaction between patients with and without CVD (hazard ratio, 0.85 [95% CI, 0.71-1.01] in patients with a history of CVD; hazard ratio, 0.86 [95% CI, 0.68-1.08] in patients without a history of CVD; P value for interaction, 0.85). The incidence of treatment-emergent adverse events was similar between treatment arms, with a low incidence of hyperkalemia-related permanent treatment discontinuation (2.3% with finerenone versus 0.8% with placebo in patients with CVD and 2.2% with finerenone versus 1.0% with placebo in patients without CVD). Conclusions: Among patients with chronic kidney disease and type 2 diabetes, finerenone reduced incidence of the composite cardiovascular outcome, with no evidence of differences in treatment effect based on preexisting CVD status. Registration: URL: https://www.clinicaltrials.gov; Unique identifier: NCT02540993.'''
# ti = '''Finerenone and Cardiovascular Outcomes in Patients With Chronic Kidney Disease and Type 2 Diabetes'''
# ti_abs = ti + " " + abstract
# sent, prob = pl_bot.make_preds_for_abstract(ti_abs)
#
# inf_bot = ev_inf_classifier.InferenceBot()
# inf_bot.predict_for_sentence(sent)
# 
# OR in one swoop...
#
# ev_bot = ev_inf_classifier.EvInfBot()
# ev_bot.infer_evidence(ti_abs)
        