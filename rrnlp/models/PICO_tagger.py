import os

import torch 
from transformers import *

import encoder 

device = "cpu"

weights_path = "weights"
weights_paths = {
    "p" : os.path.join(weights_path, "population_clf.pt"),
    "i" : os.path.join(weights_path, "interventions_clf.pt"),
    "o" : os.path.join(weights_path, "outcomes_clf.pt")
}

ids2tags = {
    "p" : {0:'pop', 1:'O'},
    "i" : {0:'intervention', 1:'O'},
    "o" : {0:'outcome', 1:'O'}
}

def get_tagging_model(element):
    # note that we assume the models were trained under I/O
    # encoding such that num_labels is 2
    model = BertForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', 
                                                        num_labels=2)

    # replace the encoder to enforce sharing by ref (and so conserving mem)
    model.bert = encoder.get_muppet()

    # load in the correct top layer weights
    clf_weights_path = weights_paths[element]
    model.classifier.load_state_dict(torch.load(clf_weights_path, 
                                        map_location=torch.device('cpu')))
    return model 


def print_labels(tokens, labels):
    all_strs, cur_str = [], []
    cur_lbl = "O"
    for token, lbl in zip(tokens, labels):
        if lbl != "O":
            cur_str.append(token)
            cur_lbl = lbl
        elif cur_lbl != "O":
            str_ = " ".join(cur_str)
            all_strs.append(str_)
            print(str_)
            cur_str = []
            cur_lbl = "O"
        
    return all_strs

def predict_for_str(model, string, id2tag, print_tokens=True, o_lbl="O"): 
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
        if print_tokens:
            print_labels(words, word_preds)

        return words_and_preds

class PICOBot:
    def __init__(self):
        self.PICO_models = {}
        for element in weights_paths: 
            print("loading model {}".format(element))
            self.PICO_models[element] = get_tagging_model(element)


    def make_preds_for_abstract(self, ti_abs):
        preds_d = {}
        for element, model in self.PICO_models.items():
            print("---- predictions for {} --- ".format(element))
            id2tag = ids2tags[element]
            preds_d[element] = predict_for_str(model, ti_abs, id2tag)
            print()
        return preds_d


'''
e.g.,

import PICO_tagger
bot = PICO_tagger.PICOBot()
an_abstract = "Introduction: To find effective and safe treatments for COVID-19, the WHO recommended to systemically evaluate experimental therapeutics in collaborative randomised clinical trials. As COVID-19 was spreading in Europe, the French national institute for Health and Medical Research (Inserm) established a transdisciplinary team to develop a multi-arm randomised controlled trial named DisCoVeRy. The objective of the trial is to evaluate the clinical efficacy and safety of different investigational re-purposed therapeutics relative to Standard of Care (SoC) in patients hospitalised with COVID-19.\n\nMethods and analysis: DisCoVeRy is a phase III, open-label, adaptive, controlled, multicentre clinical trial in which hospitalised patients with COVID-19 in need of oxygen therapy are randomised between five arms: (1) a control group managed with SoC and four therapeutic arms with re-purposed antiviral agents: (2) remdesivir + SoC, (3) lopinavir/ritonavir + SoC, (4) lopinavir/ritonavir associated with interferon (IFN)-β-1a + SoC and (5) hydroxychloroquine + SoC. The primary endpoint is the clinical status at Day 15 on the 7-point ordinal scale of the WHO Master Protocol (V.3.0, 3 March 2020). This trial involves patients hospitalised in conventional departments or intensive care units both from academic or non-academic hospitals throughout Europe. A sample size of 3100 patients (620 patients per arm) is targeted. This trial has begun on 22 March 2020. Since 5 April 2020, DisCoVeRy has been an add-on trial of the Solidarity consortium of trials conducted by the WHO in Europe and worldwide. On 8 June 2020, 754 patients have been included.\n\nEthics and dissemination: Inserm is the sponsor of DisCoVeRy. Ethical approval has been obtained from the institutional review board on 13 March 2020 (20.03.06.51744) and from the French National Agency for Medicines and Health Products (ANSM) on 9 March 2020. Results will be submitted for publication in peer-reviewed journals.\n\n"
some_preds = bot.make_preds_for_abstract(an_abstract)

another_abstract = 'Background: Current strategies for preventing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection are limited to nonpharmacologic interventions. Hydroxychloroquine has been proposed as a postexposure therapy to prevent coronavirus disease 2019 (Covid-19), but definitive evidence is lacking.\n\nMethods: We conducted an open-label, cluster-randomized trial involving asymptomatic contacts of patients with polymerase-chain-reaction (PCR)-confirmed Covid-19 in Catalonia, Spain. We randomly assigned clusters of contacts to the hydroxychloroquine group (which received the drug at a dose of 800 mg once, followed by 400 mg daily for 6 days) or to the usual-care group (which received no specific therapy). The primary outcome was PCR-confirmed, symptomatic Covid-19 within 14 days. The secondary outcome was SARS-CoV-2 infection, defined by symptoms compatible with Covid-19 or a positive PCR test regardless of symptoms. Adverse events were assessed for up to 28 days.\n\nResults: The analysis included 2314 healthy contacts of 672 index case patients with Covid-19 who were identified between March 17 and April 28, 2020. A total of 1116 contacts were randomly assigned to receive hydroxychloroquine and 1198 to receive usual care. Results were similar in the hydroxychloroquine and usual-care groups with respect to the incidence of PCR-confirmed, symptomatic Covid-19 (5.7% and 6.2%, respectively; risk ratio, 0.86 [95% confidence interval, 0.52 to 1.42]). In addition, hydroxychloroquine was not associated with a lower incidence of SARS-CoV-2 transmission than usual care (18.7% and 17.8%, respectively). The incidence of adverse events was higher in the hydroxychloroquine group than in the usual-care group (56.1% vs. 5.9%), but no treatment-related serious adverse events were reported.\n\nConclusions: Postexposure therapy with hydroxychloroquine did not prevent SARS-CoV-2 infection or symptomatic Covid-19 in healthy persons exposed to a PCR-positive case patient. (Funded by the crowdfunding campaign YoMeCorono and others; BCN-PEP-CoV2 ClinicalTrials.gov number, NCT04304053.).'
more_preds = bot.make_preds_for_abstract(another_abstract)


ok = predict_for_str(model, an_abstract)
'''
        