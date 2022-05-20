import torch
import nltk
from nltk.corpus import stopwords
from string import punctuation
from rrnlp.models.util.rct_summarize.utils import multimode
#from rrnlp.models.util.rct_summarize.directionality_classifier import EVPredictor
import json, os
stop_words = set(stopwords.words('english'))

path_to_templates = os.path.join('rrnlp', 'models', 'util', 'rct_summarize')

class TemplateGenerator():

    def __init__(self, tokenizer, device, generator):
        self.tokenizer = tokenizer
        self.device = device
        self.generator = generator
        #self.evidence_classifier =  EVPredictor()
        with open(os.path.join(path_to_templates, "templates.json"), 'r') as fp:
            self.templates = json.load(fp)
        return

    def get_direction(self, input_ids, direction = 0):
        direction = self.evidence_classifier.model_inference(input_ids, self.tokenizer)[-1]
        dir_map = {0: 'no_diff', 1: 'diff'}
        return dir_map[direction]


    def process_template(self, template, debug):
        control_keys = ['population', 'interventions', 'outcomes', 'punchline_text', 'background']
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

    def make_start_end_idx(self, outputs_and_logits, control_logit_list, start_idx =0 , get_start_idx = True):
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

    

    def shortlist_aspect_tokens(self, outputs_and_logits, control_key, debug):
        control_key_map = { 'population': 0, 'interventions': 1, 'outcomes': 2, 'punchline_text': 3, 'background' : 4}
        control_logit = control_key_map[control_key] ## the logit for current aspect of interest
        start_idx = 0 if '<s>' not in outputs_and_logits[0][0] else 1
        end_idx = self.make_start_end_idx(outputs_and_logits, [0, 1, 2], start_idx = start_idx, get_start_idx = False)
        return start_idx, end_idx


    def get_summary(self, 
        batch,
        template,  
        background_lm,
        num_beams = 6,  
        max_length = 100, 
        min_length = 5, 
        repetition_penalty = 1.0, 
        length_penalty = 1.0, 
        return_dict_in_generate = False,
        device = torch.device("cpu"), 
        temperature = 1.0, 
        do_sample = False,
        debug = True):

        input_text = ''
        for control_key, phrase in self.process_template(template, debug):
            if debug:
                    print(control_key, phrase)
            if phrase:
                    input_text += phrase 

            if control_key:
                    if control_key in ['outcomes', 'population', 'interventions']:
                        input_text += ' <%s>'%(control_key)
                    
                    input_ids = self.tokenizer(input_text, return_tensors = 'pt' )['input_ids']
                    start_id = torch.tensor([2])
                    input_ids = torch.cat([start_id, input_ids[0][1:-1]]).tolist() ## add eos token to input to the decoder followed by phrase tokens
                    input_ids = torch.tensor([input_ids]) 
                    input_ids = input_ids.to(self.device)

                    if debug:
                        print('INPUTS MADE', input_ids, input_text)
 
                    max_length = input_ids.shape[-1] + 10
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

                    
                    start_idx, end_idx = self.shortlist_aspect_tokens(list(zip(model_output_tokens, logits)), control_key, debug)
                    shortlisted_tokens = model_output_tokens[start_idx: end_idx + 1]
                    if debug:
                        print('SHORTLISTED', shortlisted_tokens)
                    template_filler = ''.join(shortlisted_tokens).strip()

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
                       
                       template_filler = ' '.join(template_filler)
                       template_filler = ' '.join([w.strip().lower() for w in template_filler.split(' ')])
                       
                       template_filler_tokens = template_filler.split(' ')
                       template_filler_non_deg = []
                       for i , w in enumerate(template_filler_tokens):
                           if w != template_filler_tokens[i-1] or i == 0:
                              if w.strip(punctuation):
                                 template_filler_non_deg.append(w)
                       template_filler = ' '.join(template_filler_non_deg)
                       if debug:
                          print('TEMPLATE FILLER', template_filler)
                       input_text += ' ' + template_filler.strip() + ' '

        if debug:                  
            print('TEMPLATE FILLED',template,  input_text)
            print('-' * 13) 
        return input_text.capitalize()

    

            
