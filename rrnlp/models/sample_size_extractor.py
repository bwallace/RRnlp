import operator 
import os 

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
word_embeddings_path = os.path.join(weights_path, "PubMed-w2v.bin")
MLP_weights_path     = os.path.join(weights_path, "sample_size_weights.pt")


def replace_n_equals(abstract_tokens):
    for j, t in enumerate(abstract_tokens):
        if "n=" in t.lower():
            # special case for sample size reporting 
            t_n = t.split("=")[1].replace(")", "") # also replace closing paren, if present
            abstract_tokens[j] = t_n 
    return abstract_tokens


class MLPSampleSize(nn.Module):
    ''' 
    The actual torch module; a very simple MLP.
    '''
    def __init__(self, n_input_features=912, n_hidden=256):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(n_input_features, n_hidden),
          nn.ReLU(),
          nn.Linear(n_hidden, 1),
          nn.Sigmoid()
        )


    def forward(self, X):
        '''
        - left/right tokens are indices corresponding to the two adjacent tokens
        - left/right PoS features are one-hot
        '''
        return self.layers(X)


class MLPSampleSizeClassifier:
    '''
    This class wraps a simple window-based (torch) MLP and bespoke 
    feature extraction functions, etc.
    '''
    def __init__(self):

        self.nlp = encoder.nlp
        # this is for POS tags
        self.PoS_tags_to_indices = {}
        self.tag_names = [u'""', u'#', u'$', u"''", u',', u'-LRB-', u'-RRB-', u'.', u':', u'ADD', u'AFX', u'BES', u'CC', u'CD', u'DT', u'EX', u'FW', u'GW', u'HVS', u'HYPH', u'IN', u'JJ', u'JJR', u'JJS', u'LS', u'MD', u'NFP', u'NIL', u'NN', u'NNP', u'NNPS', u'NNS', u'PDT', u'POS', u'PRP', u'PRP$', u'RB', u'RBR', u'RBS', u'RP', u'SP', u'SYM', u'TO', u'UH', u'VB', u'VBD', u'VBG', u'VBN', u'VBP', u'VBZ', u'WDT', u'WP', u'WP$', u'WRB', u'XX', u'``']
        for idx, tag in enumerate(self.tag_names):
            self.PoS_tags_to_indices[tag] = idx
        
        self.n_tags = len(self.tag_names)#len(self.nlp.tagger.tag_names)

        # threshold governing whether to abstain from predicting
        # this as a sample size altogether (for highest scoring 
        # integer). As always, this was definitely set in a totally
        # scientifically sound way ;).
        self.magic_threshold = 0.0205 # TODO revisit
        
        self.number_tagger = index_numbers.NumberTagger()

        self.model = MLPSampleSize()
        self.model.load_state_dict(torch.load(MLP_weights_path ))
        self.word_embeddings = load_trained_w2v_model(word_embeddings_path)
    

    def PoS_tags_to_one_hot(self, tag):
        one_hot = np.zeros(self.n_tags)
        if tag in self.PoS_tags_to_indices:
            one_hot[self.PoS_tags_to_indices[tag]] = 1.0
        else:
            pass
        return one_hot


    def featurize_for_input(self, X):
        Xv = []

        left_token_inputs, left_PoS, right_token_inputs, right_PoS, other_inputs = [], [], [], [], []

        # consider just setting to zeros?
        unk_vec = np.mean(self.word_embeddings.vectors, axis=0)

        def get_w_embedding(w): 
            try:
                return self.word_embeddings[w]
            except:
                return unk_vec

        for x in X:

            left_embeds = np.concatenate([get_w_embedding(w_i) for w_i in x["left_word"]])
            right_embeds = np.concatenate([get_w_embedding(w_i) for w_i in x["right_word"]])

            left_pos = self.PoS_tags_to_one_hot(x["left_PoS"])
            right_pos = self.PoS_tags_to_one_hot(x["left_PoS"])

            other_features = np.array(x["other_features"])

            xv = np.concatenate([left_embeds, right_embeds, left_pos, right_pos])

            Xv.append(torch.tensor(xv))
   
        return Xv 


    def predict_for_abstract(self, abstract_text):
        ''' 
        returns either the extracted sample size, or None if this cannot
        be located. 
        '''
        abstract_text_w_numbers = self.number_tagger.swap(abstract_text)
        abstract_tokens, POS_tags = tokenize_abstract(abstract_text_w_numbers, self.nlp)
        abstract_tokens = replace_n_equals(abstract_tokens)

        abstract_features, numeric_token_indices = abstract2features(abstract_tokens, POS_tags)

        X = torch.vstack(self.featurize_for_input(abstract_features)).float()
       
        
        preds = self.model(X).detach().numpy()
        most_likely_idx = np.argmax(preds)
        
        if preds[most_likely_idx] >= self.magic_threshold:
            return abstract_tokens[numeric_token_indices[most_likely_idx]]
        else:
            return None 


def load_trained_w2v_model(path):
    m = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return m

def y_to_bin(y):
    y_bin = np.zeros(len(y))
    for idx, y_i in enumerate(y):
        if y_i == "N":
            y_bin[idx] = 1.0
    return y_bin

def _is_an_int(s):
    try:
        int(s) 
        return True
    except:
        return False


def tokenize_abstract(abstract, nlp=None):

    tokens, POS_tags = [], []
    ab = nlp(abstract)
    for word in ab:
        tokens.append(word.text)
        POS_tags.append(word.tag_)
    
    return tokens, POS_tags

def abstract2features(abstract_tokens, POS_tags):

    ####
    # some of the features we use rely on 'global' info,
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

    # note that we keep track of all candidates/numbers
    # and pass this back.
    x, numeric_token_indices = [], []
    for word_idx in range(len(abstract_tokens)):
        if (_is_an_int(abstract_tokens[word_idx])):   
            numeric_token_indices.append(word_idx)         
            features = word2features(abstract_tokens, POS_tags, word_idx, all_nums_in_abstract, 
                                      years_indices, patient_indices)
            x.append(features)


    return x, numeric_token_indices


def get_window_indices(all_tokens, i, window_size):
    lower_idx = max(0, i-window_size)
    upper_idx = min(i+window_size, len(all_tokens)-1)
    return (lower_idx, upper_idx)

def word2features(abstract_tokens, POS_tags, i, all_nums_in_abstract, 
                    years_indices, patient_indices,
                    window_size_for_years=5,
                    window_size_patient_mention=4):
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
    # need to add a feature for being largest in doc??
    biggest_num_in_abstract = 0.0
    if target_num >= max(all_nums_in_abstract):
        biggest_num_in_abstract = 1.0

    # this feature encodes whether "year" or "years" is mentioned
    # within window_size_for_years tokens of the target (i)
    years_mention_within_window = 0.0
    lower_idx, upper_idx = get_window_indices(abstract_tokens, i, window_size_for_years)
    for year_idx in years_indices:
        if lower_idx < year_idx <= upper_idx:
            years_mention_within_window = 1.0
            break 

    # ditto the above, but for "patients"
    patients_mention_follows_within_window = 0.0
    _, upper_idx = get_window_indices(abstract_tokens, i, window_size_patient_mention)
    for patient_idx in patient_indices:
        if i < patient_idx <= upper_idx:
            patients_mention_follows_within_window = 1.0
            break

    target_looks_like_a_year = 0.0
    lower_year, upper_year = 1940, 2020 # totally made up.
    if lower_year <= target_num <= upper_year:
        target_looks_like_a_year = 1.0

    return {"left_word":[ll_word, l_word], # "target": target_num, 
            "right_word":[rr_word, r_word],  
            "left_PoS":l_POS, "right_PoS":r_POS, 
            "other_features":[biggest_num_in_abstract, years_mention_within_window, 
                                target_looks_like_a_year, 
                                patients_mention_follows_within_window]}




###
# on import, check if word vectors are where we expect; if not, fetch them.
###
import sys
import time
import urllib

def reporthook(count, block_size, total_size):
    # shamelessly stolen: https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

wv_url = "http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin"
if not os.path.exists(word_embeddings_path):
    import urllib 
    print(f"\nWhoops! PubMed-w2v.bin (static word vectors) not found at {word_embeddings_path}.")
    print(f"Attempting to fetch (static) PubMed word vector weights from {wv_url} ... (Will take a bit; the file is ~2gb. Maybe make a coffee.)")
    # TODO this is slow so should probably add a progress bar;
    # at present it just kinda sits there for a long time.
    
    urllib.request.urlretrieve(wv_url, word_embeddings_path, reporthook)
    if os.path.exists(word_embeddings_path):
        print("success!")
    else:
        raise Exception("Sorry; unable to download static word vectors, and will not be able to use this model.")


def example():
    from rrnlp.models import sample_size_extractor
    ss = sample_size_extractor.MLPSampleSizeClassifier()

    abstract = '''Background: World Health Organization expert groups recommended mortality trials of four repurposed antiviral drugs - remdesivir, hydroxychloroquine, lopinavir, and interferon beta-1a - in patients hospitalized with coronavirus disease 2019 (Covid-19). Methods: We randomly assigned inpatients with Covid-19 equally between one of the trial drug regimens that was locally available and open control (up to five options, four active and the local standard of care). The intention-to-treat primary analyses examined in-hospital mortality in the four pairwise comparisons of each trial drug and its control (drug available but patient assigned to the same care without that drug). Rate ratios for death were calculated with stratification according to age and status regarding mechanical ventilation at trial entry. Results: At 405 hospitals in 30 countries, 11,330 adults underwent randomization; 2750 were assigned to receive remdesivir, 954 to hydroxychloroquine, 1411 to lopinavir (without interferon), 2063 to interferon (including 651 to interferon plus lopinavir), and 4088 to no trial drug. Adherence was 94 to 96% midway through treatment, with 2 to 6% crossover. In total, 1253 deaths were reported (median day of death, day 8; interquartile range, 4 to 14). The Kaplan-Meier 28-day mortality was 11.8% (39.0% if the patient was already receiving ventilation at randomization and 9.5% otherwise). Death occurred in 301 of 2743 patients receiving remdesivir and in 303 of 2708 receiving its control (rate ratio, 0.95; 95% confidence interval [CI], 0.81 to 1.11; P = 0.50), in 104 of 947 patients receiving hydroxychloroquine and in 84 of 906 receiving its control (rate ratio, 1.19; 95% CI, 0.89 to 1.59; P = 0.23), in 148 of 1399 patients receiving lopinavir and in 146 of 1372 receiving its control (rate ratio, 1.00; 95% CI, 0.79 to 1.25; P = 0.97), and in 243 of 2050 patients receiving interferon and in 216 of 2050 receiving its control (rate ratio, 1.16; 95% CI, 0.96 to 1.39; P = 0.11). No drug definitely reduced mortality, overall or in any subgroup, or reduced initiation of ventilation or hospitalization duration. Conclusions: These remdesivir, hydroxychloroquine, lopinavir, and interferon regimens had little or no effect on hospitalized patients with Covid-19, as indicated by overall mortality, initiation of ventilation, and duration of hospital stay. (Funded by the World Health Organization; ISRCTN Registry number, ISRCTN83971151; ClinicalTrials.gov number, NCT04315948.).'''

    ss.predict_for_abstract(abstract)
