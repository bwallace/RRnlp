'''
This module responsible for attempting to extract sample sizes 
(number randomized) from abstracts. We do this by identifying
integer tokens in inputs, and then assembling bespoke feature 
vectors for each of these, and running these through a simple
MLP. This module relies on static word vectors. 
'''

import operator 
import os 
import sys
import typing 
import time
import urllib
from typing import Type, Tuple, List

import numpy as np 
import pandas as pd 

import gensim 

import torch 
from torch import nn 

import rrnlp
from rrnlp.models.util import index_numbers
from rrnlp.models import encoder 

device = rrnlp.models.device 
weights_path = rrnlp.models.weights_path
doi = rrnlp.models.files_needed['sample_size_extractor']['zenodo']
word_embeddings_path = os.path.join(weights_path, f"{doi}_PubMed-w2v.bin") # note that this is not DOI'ed - but fetched from the gensim source
MLP_weights_path     = os.path.join(weights_path, f"{doi}_sample_size_weights.pt")


def replace_n_equals(abstract_tokens: List[str]) -> List[str]:
    ''' Helper to replace "n=" occurences '''
    for j, t in enumerate(abstract_tokens):
        if "n=" in t.lower():
            # Also replace closing paren, if present
            t_n = t.split("=")[1].replace(")", "") 
            abstract_tokens[j] = t_n 
    return abstract_tokens


class MLPSampleSize(nn.Module):
    ''' A very simple MLP for sample size extraction. '''
    def __init__(self, n_input_features: int=912, n_hidden: int=256):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(n_input_features, n_hidden),
          nn.ReLU(),
          nn.Linear(n_hidden, 1),
          nn.Sigmoid()
        )


    def forward(self, X: Type[torch.Tensor]) -> Type[torch.FloatTensor]:
        ''' 
        Returns a probability that the input tokens represented by rows 
        of X are sample sizes.
        '''
        return self.layers(X)


class MLPSampleSizeClassifier:
    '''
    This class wraps a simple window-based (torch) MLP and bespoke 
    feature extraction functions, etc.
    '''
    def __init__(self):

        self.nlp = encoder.nlp
        # This is for POS tags
        self.PoS_tags_to_indices = {}
        self.tag_names = [u'""', u'#', u'$', u"''", u',', u'-LRB-', u'-RRB-', u'.', u':', u'ADD', u'AFX', u'BES', u'CC', u'CD', u'DT', u'EX', u'FW', u'GW', u'HVS', u'HYPH', u'IN', u'JJ', u'JJR', u'JJS', u'LS', u'MD', u'NFP', u'NIL', u'NN', u'NNP', u'NNPS', u'NNS', u'PDT', u'POS', u'PRP', u'PRP$', u'RB', u'RBR', u'RBS', u'RP', u'SP', u'SYM', u'TO', u'UH', u'VB', u'VBD', u'VBG', u'VBN', u'VBP', u'VBZ', u'WDT', u'WP', u'WP$', u'WRB', u'XX', u'``']
        for idx, tag in enumerate(self.tag_names):
            self.PoS_tags_to_indices[tag] = idx
        
        self.n_tags = len(self.tag_names)#len(self.nlp.tagger.tag_names)

        # Threshold governing whether to abstain from predicting
        # this as a sample size altogether (for highest scoring 
        # integer). As always, this was definitely set in a totally
        # scientifically sound way ;).
        self.magic_threshold = 0.0205 # @TODO revisit
        
        self.number_tagger = index_numbers.NumberTagger()

        self.model = MLPSampleSize()
        self.model.load_state_dict(torch.load(MLP_weights_path ))
        self.word_embeddings = load_trained_w2v_model(word_embeddings_path)
    

    def PoS_tags_to_one_hot(self, tag: str) -> Type[np.array]:
        ''' 
        Helper to map from string tags to one hot vectors encoding them.
        '''
        one_hot = np.zeros(self.n_tags)
        if tag in self.PoS_tags_to_indices:
            one_hot[self.PoS_tags_to_indices[tag]] = 1.0
        else:
            pass
        return one_hot


    def featurize_for_input(self, X: List[dict]) -> List[Type[torch.Tensor]]:
        '''
        Map from a list of dictionaries mapping features to values to a 
        torch Tensor representing the given input.
        '''
        Xv = []

        left_token_inputs, left_PoS, right_token_inputs, right_PoS, other_inputs = [], [], [], [], []

        # Consider just setting to zeros?
        unk_vec = np.mean(self.word_embeddings.vectors, axis=0)

        # Helper to grab embeddings for words where available, unk 
        # vector otherwise
        def get_w_embedding(w): 
            try:
                return self.word_embeddings[w]
            except:
                return unk_vec

        # Iterate over all instances in input, map from dictionaries of features to 
        # tensors that encode them.
        for x in X:

            left_embeds = np.concatenate([get_w_embedding(w_i) for w_i in x["left_word"]])
            right_embeds = np.concatenate([get_w_embedding(w_i) for w_i in x["right_word"]])

            left_pos = self.PoS_tags_to_one_hot(x["left_PoS"])
            right_pos = self.PoS_tags_to_one_hot(x["left_PoS"])

            other_features = np.array(x["other_features"])

            xv = np.concatenate([left_embeds, right_embeds, left_pos, right_pos])

            Xv.append(torch.tensor(xv))
   
        return Xv 


    def predict_for_ab(self, ab: dict) -> typing.Union[str, None]:
        ''' 
        Given an abstract, this returns either the extracted sample 
        size, or None if this cannot be located. 
        '''
        abstract_text = ab['ab']
        
        abstract_text_w_numbers = self.number_tagger.swap(abstract_text)
        abstract_tokens, POS_tags = tokenize_abstract(abstract_text_w_numbers, self.nlp)
        abstract_tokens = replace_n_equals(abstract_tokens)

        # Extract dictionaries of features for each token in the abstract
        abstract_features, numeric_token_indices = abstract2features(abstract_tokens, POS_tags)
        
        # If there are no numbers in the input text, then just give up
        if len(numeric_token_indices) == 0:
            return {"num_randomized": None}

        # Convert to a m x d Tensor (m = number of tokens; d = input dims)
        X = torch.vstack(self.featurize_for_input(abstract_features)).float()

        # Make prediction, retrieve associated token
        preds = self.model(X).detach().numpy()
        most_likely_idx = np.argmax(preds)
        
        # Abstain from returning a token if the *best* scoring token is beneath
        # a somewhat arbitrarily chosen threshold (since not all abstracts will
        # contain sample sizes.)
        if preds[most_likely_idx] >= self.magic_threshold:
            return {"num_randomized": abstract_tokens[numeric_token_indices[most_likely_idx]]}
        else:
            return {"num_randomized": None}


def load_trained_w2v_model(path: str) -> Type[gensim.models.keyedvectors.KeyedVectors]:
    ''' Load in and return word vectors at the given path. '''
    m = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return m

def y_to_bin(y: List[str]) -> Type[np.array]:
    y_bin = np.zeros(len(y))
    for idx, y_i in enumerate(y):
        if y_i == "N":
            y_bin[idx] = 1.0
    return y_bin

def _is_an_int(s: str) -> bool:
    try:
        int(s) 
        return True
    except:
        return False


def tokenize_abstract(abstract: str, nlp=None) -> Tuple[List[str], List[str]]:
    ''' 
    Tokenizes given abstract string, returns tokens and inferred PoS tags. 
    '''
    tokens, POS_tags = [], []
    ab = nlp(abstract)
    for word in ab:
        tokens.append(word.text)
        POS_tags.append(word.tag_)
    
    return tokens, POS_tags

def abstract2features(abstract_tokens: List[str], POS_tags: List[str]) \
        -> Tuple[List[dict], List[int]]:
    '''
    Given a tokenized input abstract (and associated list of predicted
    PoS tags), this function assembles a dictionary of artisinal features
    extracted from the inputs relevant to predicting whether the 
    constituent tokens are sample sizes or not. We consider *only* integer
    candidates as potential sample sizes, and return features for these
    as well as their indices.
    '''

    ####
    # Some of the features we use rely on 'global' info,
    # so we take a pass over the entire abstract here
    # to extract what we need:
    #   1. keep track of all numbers in the abstract
    #   2. keep track of indices where "years" mentioned
    #   3. keep track of indices where "patients" mentioned
    # the latter because years are a potential source of
    # confusion!
    years_tokens = ["years", "year"]
    patients_tokens = ["patients", "subjects", "participants"]
    all_nums_in_abstract, years_indices, patient_indices = [], [], []
    for idx, t in enumerate(abstract_tokens):
        t_lower = t.lower()

        if t_lower in years_tokens:
            years_indices.append(idx)

        if t_lower in patients_tokens:
            patient_indices.append(idx) 

        try:
            num = int(t)
            all_nums_in_abstract.append(num)
        except:
            pass

    # Note that we keep track of all candidates/numbers
    # and pass this back.
    x, numeric_token_indices = [], []
    for word_idx in range(len(abstract_tokens)):
        if (_is_an_int(abstract_tokens[word_idx])):   
            numeric_token_indices.append(word_idx)         
            features = word2features(abstract_tokens, POS_tags, word_idx, 
                                     all_nums_in_abstract, years_indices, 
                                     patient_indices)
            x.append(features)

    # Here x is a list of dictionaries encoding features (key/val pairs)
    # for all *numerical* tokens we identified — we treat these as 
    # candidate sample size tokens to be scored; the corresponding
    # indices for these candidates are stored in the second return
    # value, ie., numeric_token_indices.
    return x, numeric_token_indices


def get_window_indices(all_tokens: List[str], i: int, window_size: int)\
         -> Tuple[int, int]:
    lower_idx = max(0, i-window_size)
    upper_idx = min(i+window_size, len(all_tokens)-1)
    return (lower_idx, upper_idx)

def word2features(abstract_tokens: List[str], POS_tags: List[str], i:int, 
                  all_nums_in_abstract: List[int], years_indices: List[int],
                  patient_indices: List[int], window_size_for_years: int = 5,
                  window_size_patient_mention: int = 4) -> dict:
    '''
    Returns a dictionary of features for the token at position i, using
    the global (abstract) information provided in the given input 
    lists. 

    @TODO this function is a mess and should be rewritten.
    '''
    ll_word, l_word, r_word, rr_word = "", "", "", ""

    l_POS, r_POS   = "", ""
    t_word = abstract_tokens[i]

    if i > 1:
        ll_word = abstract_tokens[i-2].lower()
    else: 
        ll_word = "BoS"

    if i > 0:
        l_word = abstract_tokens[i-1].lower()
        l_POS  = POS_tags[i-1]
    else:
        l_word = "BoS"
        l_POS  = "XX" # i.e., unknown

    if i < len(abstract_tokens)-2:
        rr_word = abstract_tokens[i+2].lower()
    else:
        r_word = "LoS"

    if i < len(abstract_tokens)-1:
        r_word = abstract_tokens[i+1].lower()
        r_POS  = POS_tags[i+1]
    else: 
        r_word = "LoS"
        r_POS  = "XX"

    target_num = int(t_word)
    # Add a feature for being largest in document
    biggest_num_in_abstract = 0.0
    if target_num >= max(all_nums_in_abstract):
        biggest_num_in_abstract = 1.0

    # This feature encodes whether "year" or "years" is mentioned
    # within window_size_for_years tokens of the target (i)
    years_mention_within_window = 0.0
    lower_idx, upper_idx = get_window_indices(abstract_tokens, i, window_size_for_years)
    for year_idx in years_indices:
        if lower_idx < year_idx <= upper_idx:
            years_mention_within_window = 1.0
            break 

    # Ditto the above, but for "patients"
    patients_mention_follows_within_window = 0.0
    _, upper_idx = get_window_indices(abstract_tokens, i, window_size_patient_mention)
    for patient_idx in patient_indices:
        if i < patient_idx <= upper_idx:
            patients_mention_follows_within_window = 1.0
            break

    # Some adhocery (craft feature engineering!)
    target_looks_like_a_year = 0.0
    lower_year, upper_year = 1940, 2020 # totally made up.
    if lower_year <= target_num <= upper_year:
        target_looks_like_a_year = 1.0

    return {"left_word":[ll_word, l_word], 
            "right_word":[rr_word, r_word],  
            "left_PoS":l_POS, "right_PoS":r_POS, 
            "other_features":[biggest_num_in_abstract, years_mention_within_window, 
                                target_looks_like_a_year, 
                                patients_mention_follows_within_window]}


def example():
    from rrnlp.models import sample_size_extractor
    ss = sample_size_extractor.MLPSampleSizeClassifier()

    ab = '''Background: World Health Organization expert groups recommended mortality trials of four repurposed antiviral drugs - remdesivir, hydroxychloroquine, lopinavir, and interferon beta-1a - in patients hospitalized with coronavirus disease 2019 (Covid-19). Methods: We randomly assigned inpatients with Covid-19 equally between one of the trial drug regimens that was locally available and open control (up to five options, four active and the local standard of care). The intention-to-treat primary analyses examined in-hospital mortality in the four pairwise comparisons of each trial drug and its control (drug available but patient assigned to the same care without that drug). Rate ratios for death were calculated with stratification according to age and status regarding mechanical ventilation at trial entry. Results: At 405 hospitals in 30 countries, 11,330 adults underwent randomization; 2750 were assigned to receive remdesivir, 954 to hydroxychloroquine, 1411 to lopinavir (without interferon), 2063 to interferon (including 651 to interferon plus lopinavir), and 4088 to no trial drug. Adherence was 94 to 96% midway through treatment, with 2 to 6% crossover. In total, 1253 deaths were reported (median day of death, day 8; interquartile range, 4 to 14). The Kaplan-Meier 28-day mortality was 11.8% (39.0% if the patient was already receiving ventilation at randomization and 9.5% otherwise). Death occurred in 301 of 2743 patients receiving remdesivir and in 303 of 2708 receiving its control (rate ratio, 0.95; 95% confidence interval [CI], 0.81 to 1.11; P = 0.50), in 104 of 947 patients receiving hydroxychloroquine and in 84 of 906 receiving its control (rate ratio, 1.19; 95% CI, 0.89 to 1.59; P = 0.23), in 148 of 1399 patients receiving lopinavir and in 146 of 1372 receiving its control (rate ratio, 1.00; 95% CI, 0.79 to 1.25; P = 0.97), and in 243 of 2050 patients receiving interferon and in 216 of 2050 receiving its control (rate ratio, 1.16; 95% CI, 0.96 to 1.39; P = 0.11). No drug definitely reduced mortality, overall or in any subgroup, or reduced initiation of ventilation or hospitalization duration. Conclusions: These remdesivir, hydroxychloroquine, lopinavir, and interferon regimens had little or no effect on hospitalized patients with Covid-19, as indicated by overall mortality, initiation of ventilation, and duration of hospital stay. (Funded by the World Health Organization; ISRCTN Registry number, ISRCTN83971151; ClinicalTrials.gov number, NCT04315948.).'''

    ss.predict_for_ab(ab)


###
# e.g.
#
# import sample_size_extractor
# sample_size_bot = sample_size_extractor.MLPSampleSizeClassifier()
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
# sample_size = sample_size_bot.predict_for_doc(ti_abs) 
