import os

import torch 
from transformers import *

import encoder 

device = "cpu"

clf_weights_path = os.path.join("weights", "RoB_overall_abs_clf.pt")
# In the RoB case we load in some task-specific weights
shared_encoder_weights_path = os.path.join("weights", "RoB_encoder_custom.pt")


def get_RoB_model():
    # note that we assume the models were trained under I/O
    # encoding such that num_labels is 2
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', 
                                                        num_labels=2)


    # read in encoder: a mix of shared weights and custom
    custom_encoder_layers = torch.load(shared_encoder_weights_path, 
                                        map_location=torch.device(device))

    encoder.load_encoder_layers(model.bert, encoder.get_muppet(), custom_encoder_layers)

    # load in the correct top layer weights
    model.classifier.load_state_dict(torch.load(clf_weights_path, 
                                        map_location=torch.device(device)))
    return model 




class AbsRoBBot:
    def __init__(self):
        self.RoB_model = get_RoB_model()
        self.RoB_model.eval()

    def predict_for_doc(self, ti_and_abs): 
        
        x = encoder.tokenize(ti_and_abs, is_split_into_words=False)

        with torch.no_grad():
            
            x_input_ids = torch.tensor(x['input_ids']).to(device).unsqueeze(dim=0)
            attention_mask= torch.tensor(x['attention_mask']).to(device).unsqueeze(dim=0)
            
            logits = self.RoB_model(x_input_ids, attention_mask=attention_mask)['logits'].cpu()
            probs  = torch.nn.functional.softmax(logits, dim=1).numpy()
            #import pdb; pdb.set_trace()
            prob_low_risk = probs[0][1]
            return prob_low_risk

    def make_preds_for_abstract(self, ti_and_abs):
        self.predict_for_doc(ti_and_abs)


###
# e.g.
#
# import RoB_classifier
# RoB_bot = RoB_classifier.AbsRoBBot()
#
# title    = '''Repurposed Antiviral Drugs for Covid-19 - Interim WHO Solidarity Trial Results'''
# abstract = '''Background: World Health Organization expert groups recommended mortality trials of four repurposed antiviral drugs - remdesivir, hydroxychloroquine, lopinavir, and interferon beta-1a - in patients hospitalized with coronavirus disease 2019 (Covid-19). Methods: We randomly assigned inpatients with Covid-19 equally between one of the trial drug regimens that was locally available and open control (up to five options, four active and the local standard of care). The intention-to-treat primary analyses examined in-hospital mortality in the four pairwise comparisons of each trial drug and its control (drug available but patient assigned to the same care without that drug). Rate ratios for death were calculated with stratification according to age and status regarding mechanical ventilation at trial entry. Results: At 405 hospitals in 30 countries, 11,330 adults underwent randomization; 2750 were assigned to receive remdesivir, 954 to hydroxychloroquine, 1411 to lopinavir (without interferon), 2063 to interferon (including 651 to interferon plus lopinavir), and 4088 to no trial drug. Adherence was 94 to 96% midway through treatment, with 2 to 6% crossover. In total, 1253 deaths were reported (median day of death, day 8; interquartile range, 4 to 14). The Kaplan-Meier 28-day mortality was 11.8% (39.0% if the patient was already receiving ventilation at randomization and 9.5% otherwise). Death occurred in 301 of 2743 patients receiving remdesivir and in 303 of 2708 receiving its control (rate ratio, 0.95; 95% confidence interval [CI], 0.81 to 1.11; P = 0.50), in 104 of 947 patients receiving hydroxychloroquine and in 84 of 906 receiving its control (rate ratio, 1.19; 95% CI, 0.89 to 1.59; P = 0.23), in 148 of 1399 patients receiving lopinavir and in 146 of 1372 receiving its control (rate ratio, 1.00; 95% CI, 0.79 to 1.25; P = 0.97), and in 243 of 2050 patients receiving interferon and in 216 of 2050 receiving its control (rate ratio, 1.16; 95% CI, 0.96 to 1.39; P = 0.11). No drug definitely reduced mortality, overall or in any subgroup, or reduced initiation of ventilation or hospitalization duration. Conclusions: These remdesivir, hydroxychloroquine, lopinavir, and interferon regimens had little or no effect on hospitalized patients with Covid-19, as indicated by overall mortality, initiation of ventilation, and duration of hospital stay. (Funded by the World Health Organization; ISRCTN Registry number, ISRCTN83971151; ClinicalTrials.gov number, NCT04315948.).'''
# ti_abs   = title + " " + abstract
#
# pred_low_RoB = RoB_bot.predict_for_doc(ti_abs) 
        