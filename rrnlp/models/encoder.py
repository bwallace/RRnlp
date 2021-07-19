
'''
The main purpose of this is to hold a single instance of a muppet for 
other modules to reference and share.
'''
import os
from collections import OrderedDict
from typing import Type, Tuple, List

import torch
import transformers 
transformers.logging.set_verbosity_error()
from transformers import *

import numpy as np 

import spacy 

import rrnlp 
spacy_weights_path = os.path.join(rrnlp.models.weights_path, 
                                   "en_core_sci_sm-0.4.0", "en_core_sci_sm", 
                                   "en_core_sci_sm-0.4.0")
nlp = spacy.load(spacy_weights_path)

# Change as appropriate!
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
muppet = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
#tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
#muppet = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

MAX_LEN = 512 

# Freeze encoder layers by default.
for param in muppet.parameters():
    param.requires_grad = False

def get_muppet() -> Type[BertModel]:
    return muppet

def tokenize(texts: List[str], is_split_into_words: bool=True):
    '''
    Assumes texts is a list of texts that have been **split into words**, like:

        [['Impact', 'of', 'early', 'mobilization', 'on', 'glycemic', ... ], 
         ['Results', 'of', 'the', 'EICESS-92', ...]
        ]

    Unless this arg is False, in which case assumes these are just vanilla text 
    inputs.
    '''
    if is_split_into_words:
        return tokenizer(texts, is_split_into_words=True, 
                        return_offsets_mapping=True, padding=True, 
                        truncation=True, max_length=MAX_LEN)
    
    return tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN)


'''
Helper methods for accessing/unfreezing layers.
'''
def get_top_k_BERT_layers(bert_inst: Type[BertModel], k:int, 
                            n_encoder_layers:int=12) -> dict:
    '''
    Return top $k$ layer parameters from the given (Bert) encoder
    in a parameters dictionary.
    '''
    layer_indices = [n_encoder_layers-j for j in range(1, k+1)]
    layer_d = OrderedDict()

    for params_name, param in bert_inst.named_parameters():
        if "encoder.layer" in params_name and int(params_name.split(".")[2]) in layer_indices:
            layer_d[params_name] = param

    return layer_d

def unfreeze_last_k_layers(bert_inst: Type[BertModel], k: int, 
                            n_encoder_layers:int =12) -> None:
    ''' Unfreezes the top k layers; modifies given BERT instance. '''
    encoder_layers_to_unfreeze = get_top_k_BERT_layers(bert_inst, k)
    for layer_name, layer_params in encoder_layers_to_unfreeze.items():
        layer_params.requires_grad = True  


def load_encoder_layers(bespoke_muppet: Type[BertModel], 
                        shared_muppet: Type[BertModel], 
                        custom_layers: dict):
    ''' 
    Update the target ('bespoke') BERT (or similar) encoder (first arg) such 
    that it will comprise model parameter values equal to whatever is in 
    "custom layers" for all layers that this StateDict contains, and values
    equal to those in the 'shared muppet' (another BERT or similar) otherwise.
    Modifies in-place (by val); returns None.
    '''
    updated_sd = bespoke_muppet.state_dict()

    for layer_name, layer_params in bespoke_muppet.state_dict().items():

        if layer_name in custom_layers.keys():
            updated_sd[layer_name] = custom_layers[layer_name]
        else:
            updated_sd[layer_name] = shared_muppet.state_dict()[layer_name]
   
    bespoke_muppet.load_state_dict(updated_sd)

    

