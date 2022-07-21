import torch
import nltk
from collections import Counter
from itertools import groupby
#from inference.directionality_classifier.predictor_agg import EVPredictor
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from string import punctuation


def get_templates():
    templates = {
        "diff": [ "intervention__ appears to have a beneficial effect on __outcome__ in patients with __population__"
        ],
        "no_diff": [ " There is no evidence that __intervention__ has an effect on __outcome__ in patients with __population__"
        ]}
    return templates

def  multimode(l):
   freqs = groupby(Counter(l).most_common(), lambda x:x[1])
   return [val for val,count in next(freqs)[1]]

def process_template(template, debug):
    control_keys = ['population', 'intervention', 'outcome', 'punchline_text', 'background']
    phrases = template.split('__')
    if debug:
        print(phrases)
    phrases = [each.strip() for each in phrases]

    for phrase in phrases:
        control_key = None
        if phrase in control_keys:
            control_key = phrase
            yield control_key, None
        else:
            yield control_key, phrase



def  multimode(l):
    freqs = groupby(Counter(l).most_common(), lambda x:x[1])
    return [val for val,count in next(freqs)[1]]

def check_logits_tokens(logits, control_logits_list):
    shortlist = len(list(set(logits).intersection(control_logits_list)))
    return shortlist == len(logits)


def make_start_end_idx(outputs_and_logits, control_logit_list, start_idx =0 , get_start_idx = True):
    prev_word_idx = 0
    prev_word_logits = []
    for i, outputs in enumerate(outputs_and_logits):
        o = outputs[0]
        l = outputs[1]

        if (o.strip() != o or not(o.strip('.')) or '</s>' in o) and prev_word_logits:

            mode_pw = multimode(prev_word_logits)
            mode_pw_val = [w for w in mode_pw if w in control_logit_list]
            if get_start_idx:
                if mode_pw_val and outputs_and_logits[prev_word_idx][0] != '<s>':
                    start_idx = prev_word_idx
                    return start_idx
            else:
                #print(o, l, mode_pw_val, prev_word_idx)
                if not mode_pw_val or l == 4.0:
                    #print(i)
                    if i > start_idx + 1:
                        end_idx = prev_word_idx - 1
                        #print(end_idx)
                        return end_idx

                if i == len(outputs_and_logits) - 1 :

                        if l in control_logit_list:
                            return i
                        else:
                            return i - 1


            prev_word_idx = i
            prev_word_logits = [l]

        else:
            prev_word_logits.append(l)
            if i == len(outputs_and_logits) - 1:
                mode_pw = multimode(prev_word_logits)
                mode_pw_val = [w for w in mode_pw if w in control_logit_list]
                if mode_pw_val:
                    return i
                return prev_word_idx - 1

    return start_idx if get_start_idx else start_idx + 1


def shortlist_aspect_tokens(outputs_and_logits, control_key, debug):
    control_key_map = { 'population': 0, 'intervention': 1, 'outcome': 2, 'punchline_text': 3, 'background' : 4}
    control_logit = control_key_map[control_key] ## the logit for current aspect of interest
    start_idx = make_start_end_idx(outputs_and_logits, [control_logit], start_idx =0 , get_start_idx = True)
    #start_idx = 0 if '<s>' not in outputs_and_logits[0][0] else 1
    end_idx = make_start_end_idx(outputs_and_logits, [0,1,2], start_idx = start_idx, get_start_idx = False)
    #print('START AND END', start_idx, end_idx)
    #print(''.join([each[0] for each in outputs_and_logits[start_idx:end_idx + 1]]))
    return start_idx, end_idx


class TemplateGenerator():

    def __init__(self, tokenizer, device, generator):
        self.tokenizer = tokenizer
        self.device = device
        self.generator = generator
        self.templates = get_templates()
#         #self.evidence_classifier =  EVPredictor()
#         with open('/home/ramprasad.sa/structured_summarization/inference/templates_no_diff.txt', 'r') as fp:
#                 self.templates_no_diff = fp.readlines()
#         with open('/home/ramprasad.sa/structured_summarization/inference/templates_diff.txt', 'r') as fp:
#                 self.templates_diff = fp.readlines()
        return

    def get_templates(self, direction):
        if direction == 'diff':
            return self.templates['diff'][0]
        else:
            return self.templates['no_diff'][0]

    '''def get_templates(self, input_ids, direction = 0):
        #direction = self.evidence_classifier.model_inference(input_ids, self.tokenizer)
        direction = 0
        print('DIRECTION', direction)
        no_diff = 0
        diff = 1 
        if False:
            with open('/home/ec2-user/structured_summarization/inference/templates_no_diff.txt', 'r') as fp:
                templates = fp.readlines()
        else:
            with open('/home/ec2-user/structured_summarization/inference/templates_diff.txt', 'r') as fp:
                templates = fp.readlines()
        
        return templates'''

    def get_summary(self, 
        batch,
        template,  
        background_lm = False,
        num_beams = 3,  
        max_length = 20, 
        min_length = 5, 
        repetition_penalty = 1.0, 
        length_penalty = 1.0, 
        return_dict_in_generate = False,
        device = torch.device("cpu"), 
        temperature = 1.0, 
        do_sample = False,
        debug = True):
        logits_map = { 'population': 0, 'intervention': 1, 'outcome': 2, 'punchline_text': 3, 'background' : 4}
        
        filled_templates_logits = [4]
        filled_templates = []
        filled_templates_outputs = [0]
        def remove_degeneration(template_filler):
            
            template_filler_processed = [template_filler.split(' ')[0] if template_filler.split(' ')[0] not in stop_words else '']
            for each in template_filler.split(' ')[1:]:
                        
                if template_filler_processed and (each != template_filler_processed[-1]):
                    template_filler_processed.append(each)
            
            if template_filler_processed[-1] in stop_words:
                template_filler_processed[:-1]
            template_filler = ' '.join(template_filler_processed).strip()
            return template_filler
        

        input_text = ''
        for control_key, phrase in process_template(template, debug):
            if debug:
                    print(control_key, phrase)
            if phrase:
                    input_text += ' ' + phrase 
                    input_text_tokens = self.tokenizer(' ' + phrase, return_tensors = 'pt' )['input_ids'].tolist()[0][1:-1]
                    input_logits = [4] * len(input_text_tokens)
                    filled_templates_outputs += input_text_tokens
                    filled_templates_logits += input_logits 
                       

            if control_key:
                    input_ids = self.tokenizer(input_text, return_tensors = 'pt' )['input_ids']
                    start_id = torch.tensor([2])
                    input_ids = torch.cat([start_id, input_ids[0][1:-1]]).tolist() ## add eos token to input to the decoder followed by phrase tokens
                    input_ids = torch.tensor([input_ids]) 
                    input_ids = input_ids.to(self.device)

                    if debug:
                        print('INPUTS MADE', input_ids, input_text)
 
                    max_length = input_ids.shape[-1] + 20
                    outputs, logits = self.generator.generate(batch, num_beams = num_beams,  max_length = max_length, min_length = 5, \
                                        repetition_penalty = repetition_penalty, length_penalty = length_penalty, \
                                            return_dict_in_generate = False, control_key = control_key,  \
                                                background_lm = background_lm, device = device, \
                                                temperature = temperature, do_sample = do_sample, 
                                                decoder_input_ids = input_ids, early_stopping = True) 
                    logits = logits.tolist()
                    input_ids_len = input_ids.shape[-1] ## get number of  input tokens 
                    outputs = outputs[:, input_ids_len:] ## get new outputs that follow input tokens
                    logits = logits[input_ids_len:]

                    ''' for debugging'''
                    model_output_tokens = [self.tokenizer.decode(w) for w in outputs[0]]

                    if debug:
                        print(list(zip(outputs[0], logits)))
                        print('OUTPUTS AND LOGITS', list(zip(model_output_tokens, logits)))
                        print('RETURNED', self.tokenizer.decode(outputs[0]))

                    #word_ids_map = make_token_word_ids(outputs.tolist(),logits, tokenizer )
                    start_idx, end_idx = shortlist_aspect_tokens(list(zip(model_output_tokens, logits)), control_key, debug)
                    shortlisted_tokens = model_output_tokens[start_idx: end_idx + 1]
                    if debug:
                        print('SHORTLISTED', shortlisted_tokens)
                    template_filler = ''.join(shortlisted_tokens).strip()
                    template_filler = remove_degeneration(template_filler)
                    
                    
                    #template_filler = self.tokenizer.decode(shortlisted_tokens) if shortlisted_tokens else ''
                    #template_filler = template_filler.split('.')[0].strip()
                    if template_filler:
                
                       '''include last word only if not in pos of list,
                        improve to include words where even if in pos list is followed by other desc pos'''
                       template_filler_tokens = nltk.word_tokenize(template_filler)
                       template_filler_tags = nltk.pos_tag(template_filler_tokens)
                       template_filler = [w[0].lower() for w in template_filler_tags[:-1]]
                       
                       template_filler_lw = template_filler_tags[-1][0] if \
                           (template_filler_tags[-1][0] not in stop_words) \
                               else ' '
                       template_filler += [template_filler_lw]
                       template_filler = ' '+ ' '.join(template_filler)
                    
                       if debug:
                          print('TEMPLATE FILLER', template_filler)
                        
                       template_text_tokens = self.tokenizer(template_filler, return_tensors = 'pt' )['input_ids'].tolist()[0][1:-1]
                       template_logits = [logits_map[control_key]] * len(template_text_tokens)
                       
                       filled_templates_outputs += template_text_tokens
                       filled_templates_logits += template_logits
                    
                       filled_templates.append(template_filler.strip())
                       input_text += template_filler
                   
        
        filled_templates_outputs.append(2)
        filled_templates_logits.append(4)
        
        print('TEMPLATE FILLED',template,  input_text)
        
        return filled_templates_outputs, filled_templates_logits

    

            
