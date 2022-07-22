
from transformers import LEDTokenizer, LEDForConditionalGeneration
from rrnlp.models.RCT_summarization_model import LEDForDataToTextGeneration_MultiLM
from rrnlp.models.util.rct_summarize.model_inference import Data2TextGenerator
from rrnlp.models.util.rct_summarize.model_template_inference import TemplateGenerator
import pytorch_lightning as pl
import torch
import rrnlp
from rrnlp.models import ev_inf_classifier
from rrnlp.models.util.rct_summarize.utils import _tie_decoder_weights, load_multilm_layers, load_vanilla_layers
from collections import Counter
import os

weights_path = rrnlp.models.weights_path
templates_path = os.path.join("util", "rct_summarize")
device = 'cpu' 


additional_special_tokens = ['<population>', '</population>',
                            '<interventions>', '</interventions>',
                            '<outcomes>', '</outcomes>',
                            '<punchline_text>', '</punchline_text>',
                            '<study>', '</study>', "<sep>"]



class VanillaSummaryBot():
    
    def __init__(self, max_input_len = 16000):
            self.max_input_len = max_input_len
            self.model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
            self.tokenizer =  LEDTokenizer.from_pretrained("allenai/led-base-16384", bos_token="<s>",
                                                            eos_token="</s>",
                                                            pad_token = "<pad>")
            self.tokenizer.add_tokens(additional_special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            doi = rrnlp.models.files_needed['RCT_summarize_vanilla']['zenodo']
            model_weight_file = os.path.join(weights_path, f"{doi}_refined_state_dict_vanilla.ckpt")
            pretrained_weights = torch.load(model_weight_file, map_location=torch.device('cpu') )
            pretrained_weights = load_vanilla_layers(pretrained_weights, self.model)
            
            self.model.load_state_dict(pretrained_weights)
            
    def process_spans(self, data , ev_bot):
        
        def tokenize(snippet):        
            encoded_dict = self.tokenizer(
            snippet,
            max_length=self.max_input_len,
            padding="max_length" ,
            truncation=True,
            return_tensors="pt",
            )
            return encoded_dict
    
        def strip_spaces(lst):
                lst_stripped = []
                for each in lst:
                    each = ' '.join([w.strip() for w in each.split(' ') if w.strip()])
                    lst_stripped.append(each)
                return lst_stripped

        def make_spans(spans, span_key):
            spans = ' <sep> '.join(spans)
            spans = '<%s> '%(span_key) + spans + ' </%s> '%(span_key)

            return spans

        all_data = []
        for each in data:
            if 'population' in each:
                pop = make_spans(strip_spaces(each['population']), 'population')
            if 'interventions' in each:
                inter = make_spans(strip_spaces(each['interventions']), 'interventions')
            if 'outcomes' in each:
                out = make_spans(strip_spaces(each['outcomes']), 'outcomes')
            if 'punchline_text' in each:
                ti_ab = {'ti': each['ti'], 'ab': each['punchline_text']}
                p_effect = ev_bot.predict_for_ab(ti_ab)['effect']
                ptext = make_spans(strip_spaces([each['punchline_text'], p_effect]), 'punchline_text')

            all_data.append(' '.join([pop, inter, out, ptext]))
        all_data = ['<study> ' + each + ' </study>' for each in all_data]
        all_data = ' '.join(all_data)
        print(all_data[:100])
        encoded_dict = tokenize(all_data)
        return encoded_dict['input_ids'], encoded_dict['attention_mask']
    
    
    def summarize(self, data, ev_bot): 
        input_ids, attn_mask = self.process_spans(data, ev_bot)
        print(input_ids.shape)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask = global_attention_mask.to(torch.device('cpu'))
        generated_ids = self.model.generate(
                    input_ids,
                    attention_mask = attn_mask,
                    global_attention_mask=global_attention_mask,
                    use_cache=True,
                    decoder_start_token_id = self.tokenizer.pad_token_id,
                    num_beams= 3,
                    min_length = 3,
                    max_length = 50,
                    early_stopping = True,
                    no_repeat_ngram_size = 3
            )
        model_output = ' '.join([self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids])
        model_output = ' '.join([w for w in model_output.split(' ') if w not in additional_special_tokens])
        
        return {'summary': model_output, 'aspect_indices': [] }
            
            
        

# class VanillaSummaryBot():
    
#     def __init__(self, max_input_len = 307):
        
    
class StructuredSummaryBot():
    def __init__(self, max_input_len = 3072, background_lm = False):
        self.max_input_len = max_input_len
        self.background_lm = background_lm
        self.logit_map = {0: 'population', 1: 'interventions', 2: 'outcomes', 3: 'punchline_text', 4: 'other'}
        self.dir_map =  {'— no diff': 'no_diff', '↓ sig. decrease': 'diff', '↑ sig. increase': 'diff'}
        
        self.tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384", bos_token="<s>",
                                                            eos_token="</s>",
                                                            pad_token = "<pad>")
        self.tokenizer.add_tokens(additional_special_tokens)
        self.model = LEDForDataToTextGeneration_MultiLM.from_pretrained('allenai/led-base-16384')
        self.model.resize_token_embeddings(len(self.tokenizer)) 
        self.model._make_decoders(4, background_lm)
        self.model.to(torch.device(device))

        doi = rrnlp.models.files_needed['RCT_summarizer_structured']['zenodo']
        model_weight_file = os.path.join(weights_path, f"{doi}_refined_state_dict_multilm.ckpt")
        pretrained_weights = torch.load(model_weight_file, map_location=torch.device('cpu') )
        pretrained_weights = load_multilm_layers(pretrained_weights, self.model)
        self.model.load_state_dict(pretrained_weights)

        self.generator = Data2TextGenerator(self.model, self.tokenizer)
        #self.template_generator = TemplateGenerator(self.tokenizer, torch.device(device), self.generator)
        #self.templates = self.template_generator.templates
        #self.ev_bot = ev_inf_classifier.EvInfBot()
        
        

#     def template_summary(self, batch):

#         def get_temp_outputs(templates):
#             templates_generated_text = []
#             for template in templates[:1]:
#                 generated_text = self.template_generator.get_summary(batch, template , background_lm = True,
#                                     num_beams = 4, max_length = 400, min_length = 13, 
#                                     repetition_penalty=1.0, length_penalty=1.0,
#                                     return_dict_in_generate=False, device = device,
#                                     temperature = 0.9, do_sample = False, debug = False)
#                 generated_text = ' '.join([w for w in generated_text.split(' ') if w not in additional_special_tokens])
#                 print(generated_text) 

#                 templates_generated_text.append(generated_text)
#             return templates_generated_text 

#         ptext_input_ids= batch[6]
#         model_outputs_temp_diff = get_temp_outputs(self.templates['diff'])
#         model_outputs_temp_nodiff = get_temp_outputs(self.templates['no_diff'])
#         return  model_outputs_temp_diff, model_outputs_temp_nodiff

        
    def _get_logit_mapped(self, logits):
        logits = logits.tolist()
        logits = [self.logit_map[each] for each in logits]
        return logits

    def _get_summary_direction(self, model_output):
        ti_ab = {'ti': '', 'ab': """%s"""%(model_output)}
        
        pred_direction = self.dir_map[self.ev_bot.predict_for_ab(ti_ab)['effect']]
        return pred_direction
    
    def run_tokenizer(self, spans, span_key):
        def tokenize(snippet):        
            encoded_dict = self.tokenizer(
            snippet,
            max_length=self.max_input_len,
            padding="max_length" ,
            truncation=True,
            return_tensors="pt",
            )
            return encoded_dict

        spans = [ ' <sep> '.join(each) for each in spans]
        spans = ['<study> <%s> '%(span_key) + each + ' </%s> </study>'%(span_key) for each in spans]
        spans = ' '.join(spans)
        print('SPAN',spans)
        encoded_dict = tokenize(spans)
        return encoded_dict['input_ids'], encoded_dict['attention_mask']
    
    def process_spans(self, data, ev_bot):
        def strip_spaces(lst):
            lst_stripped = []
            for each in lst:
                each = ' '.join([w.strip() for w in each.split(' ') if w.strip()])
                lst_stripped.append(each)
            return lst_stripped
        
        p_spans = []
        i_spans = []
        o_spans = []
        ptext_spans = []
        
        
        for each in data:
            if 'population' in each:
                p_spans.append(strip_spaces(each['population']))
            if 'interventions' in each:
                i_spans.append(strip_spaces(each['interventions']))
            if 'outcomes' in each:
                o_spans.append(strip_spaces(each['outcomes']))
            if 'punchline_text' in each:
                ti_ab = {'ti': each['ti'], 'ab': each['punchline_text']}
                p_effect = ev_bot.predict_for_ab(ti_ab)['effect']
                ptext_spans.append(strip_spaces([each['punchline_text'], p_effect]))
            
        p_input_ids, p_attn_masks = self.run_tokenizer(p_spans, 'population')
        i_input_ids, i_attn_masks = self.run_tokenizer(i_spans, 'interventions')
        o_input_ids, o_attn_masks = self.run_tokenizer(o_spans, 'outcomes')
        ptext_input_ids, ptext_attn_masks = self.run_tokenizer(ptext_spans, 'punchline_text')
        return p_input_ids, p_attn_masks, i_input_ids, i_attn_masks, o_input_ids, o_attn_masks, ptext_input_ids, ptext_attn_masks
    
    
    def summarize_template(self, data, ev_bot):
        batch = self.process_spans(data, ev_bot)
        
        
    def process_word_logits(self, outputs, logits):
        def process_logits(word_logits, l):
            if not word_logits:
                word_logits = [l]
            word_logits_count = dict(Counter(word_logits))
            shortlisted_aspects = [k for k,v in word_logits_count.items() if v ==  max(word_logits_count.values())] 
            word_logit = shortlisted_aspects[0]
            if len(shortlisted_aspects) > 1:
                if word_logits[0] in shortlisted_aspects:
                    word_logit = word_logits[0]
            return word_logit
        
        all_w_logits = []
        w = []
        word_logits = []
        #print(len(outputs[0]), len(logits))
        #print(self.tokenizer.decode(outputs[0], skip_special_tokens = True))
        
        for t, l in list(zip(outputs[0], list(logits))):
            t = self.tokenizer.decode(t, skip_special_tokens = True)
            #print('TRUE', t, l)
            if t not in additional_special_tokens:
            
                t = t.split('<')[0]
                if t != t.strip():
                    word_logit = process_logits(word_logits, l)
                    
                    all_w_logits.append((''.join(w), word_logit))
                    w = [t]
                    word_logits = [l]

                else:
                    w.append(t)
                    word_logits.append(l)
            else:
                all_w_logits.append((''.join(w), process_logits(word_logits, l)))
                w = []
                word_logits = []
        
        if w:
            all_w_logits.append((''.join(w), process_logits(word_logits, l)))
        
        #print(all_w_logits)
        model_output = [each[0].strip() for each in all_w_logits[1:]]
        logits = [each[1] for each in all_w_logits[1:]]
        summary = ' '.join(model_output)
        
        return summary, logits
        
    def summarize(self, data, ev_bot):
        # TODO: replace with real generated summary
        # currently this is just a dummy summarizer that spits out the punchline of the first study
        batch = self.process_spans(data, ev_bot)
        print('summarizing ...')
        outputs, logits = self.generator.generate(batch, num_beams = 3,  max_length = 70, min_length = 3, \
            repetition_penalty = 1.0, length_penalty = 1.0, early_stopping = True, \
                return_dict_in_generate = False, control_key = None, no_repeat_ngram_size = 3, \
                    background_lm = False, device = torch.device(device))
        
        logits = self._get_logit_mapped(logits)
    
        summary, logits  = self.process_word_logits(outputs, logits)
        
#         print('SUMMARY', summary, summary.split(' '))
#         print('LOGITS', logits)
        
        
        return {'summary': summary, 'aspect_indices': logits }
    
    
    def summarize_template(self, data, vanilla_summary, direc, ev_bot):
        batch = self.process_spans(data, ev_bot)
        TempGen = TemplateGenerator(self.tokenizer, torch.device('cpu'), self.generator)
        template = TempGen.get_templates(direc)
        
        outputs, logits = TempGen.get_summary(batch, template )
        print(outputs, logits)
        
        logits = self._get_logit_mapped(torch.tensor(logits))
        summary, logits  = self.process_word_logits([outputs], logits)
        #logits = self._get_logit_mapped(logits)
        
        return {'summary': summary, 'aspect_indices': logits }
