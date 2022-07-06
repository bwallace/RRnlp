
from transformers import LEDTokenizer
from rrnlp.models.RCT_summarization_model import LEDForDataToTextGeneration_MultiLM_Background
from rrnlp.models.util.rct_summarize.model_inference import Data2TextGenerator
#`from rrnlp.models.util.rct_summarize.model_template_inference import TemplateGenerator
import pytorch_lightning as pl
import torch
import rrnlp
from rrnlp.models import ev_inf_classifier
from collections import Counter
import os

weights_path = rrnlp.models.weights_path
doi = rrnlp.models.files_needed['RCT_summarizer']['zenodo']

model_weight_file = os.path.join(weights_path, f"{doi}_refined_state_dict.ckpt")

device = 'cpu' 


additional_special_tokens = ['<population>', '</population>',
                            '<interventions>', '</interventions>',
                            '<outcomes>', '</outcomes>',
                            '<punchline_text>', '</punchline_text>',
                            '<study>', '</study>', "<sep>"]

def load_layers(saved_layers, model):
    model_updated_state_dict = model.state_dict()

    for layer_name, layer_params in model.state_dict().items():
        saved_layer_name = 'model.'+layer_name

        if saved_layer_name  in saved_layers.keys():
            model_updated_state_dict[layer_name] = saved_layers[saved_layer_name]
            #print('FOUND', saved_layer_name)
        else:
            if 'decoder' in layer_name:
                decoder_key = layer_name.split('.')
                shared_decoder_key =  ['decoder'] + decoder_key[1:]
                shared_decoder_key = '.'.join(shared_decoder_key)
                saved_layer_name = 'model.'+shared_decoder_key
                model_updated_state_dict[layer_name]  = saved_layers[saved_layer_name]
                print('SHARED', layer_name, saved_layer_name)
            else:
                print(layer_name)
    return model_updated_state_dict


class StructuredSummaryBot():
    def __init__(self, max_input_len = 3072):
        
        self.logit_map = {0: 'population', 1: 'interventions', 2: 'outcomes', 3: 'punchline_text', 4: 'other'}
        self.dir_map =  {'— no diff': 'no_diff', '↓ sig. decrease': 'diff', '↑ sig. increase': 'diff'}
        
        self.tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384", bos_token="<s>",
                                                            eos_token="</s>",
                                                            pad_token = "<pad>")
        self.tokenizer.add_tokens(additional_special_tokens)


        self.model = LEDForDataToTextGeneration_MultiLM_Background.from_pretrained('allenai/led-base-16384')
        self.model.resize_token_embeddings(len(self.tokenizer)) 
        self.model._make_decoders(4)
        self.model.to(torch.device(device))

        pretrained_weights = torch.load(model_weight_file, map_location=torch.device('cpu') )
        pretrained_weights = load_layers(pretrained_weights, self.model)
        self.model.load_state_dict(pretrained_weights)

        self.generator = Data2TextGenerator(self.model, self.tokenizer)
        #self.template_generator = TemplateGenerator(self.tokenizer, torch.device(device), self.generator)
        #self.templates = self.template_generator.templates
        #self.ev_bot = ev_inf_classifier.EvInfBot()
        
        

    def template_summary(self, batch):

        def get_temp_outputs(templates):
            templates_generated_text = []
            for template in templates[:1]:
                generated_text = self.template_generator.get_summary(batch, template , background_lm = True,
                                    num_beams = 4, max_length = 400, min_length = 13, 
                                    repetition_penalty=1.0, length_penalty=1.0,
                                    return_dict_in_generate=False, device = device,
                                    temperature = 0.9, do_sample = False, debug = False)
                generated_text = ' '.join([w for w in generated_text.split(' ') if w not in additional_special_tokens])
                print(generated_text) 

                templates_generated_text.append(generated_text)
            return templates_generated_text 

        ptext_input_ids= batch[6]
        model_outputs_temp_diff = get_temp_outputs(self.templates['diff'])
        model_outputs_temp_nodiff = get_temp_outputs(self.templates['no_diff'])
        return  model_outputs_temp_diff, model_outputs_temp_nodiff

        
    def _get_logit_mapped(self, logits):
        logits = logits.tolist()
        logits = [self.logit_map[each] for each in logits]
        return logits

    def _get_summary_direction(self, model_output):
        ti_ab = {'ti': '', 'ab': """%s"""%(model_output)}
        
        pred_direction = self.dir_map[self.ev_bot.predict_for_ab(ti_ab)['effect']]
        return pred_direction
        
    def summarize(self, batch):
        # TODO: replace with real generated summary
        # currently this is just a dummy summarizer that spits out the punchline of the first study
        print('summarizing ...')
        outputs, logits = self.generator.generate(batch, num_beams = 2,  max_length = 10, min_length = 3, \
            repetition_penalty = 1.0, length_penalty = 1.0, early_stopping = True, \
                return_dict_in_generate = False, control_key = None, no_repeat_ngram_size = 3, \
                    background_lm = True, device = torch.device(device))
        
        logits = self._get_logit_mapped(logits)
        
        all_w_logits = []
        w = []
        word_logits = []
        print(len(outputs[0]), len(logits))
        print(self.tokenizer.decode(outputs[0], skip_special_tokens = True))
        
        def process_logits(word_logits):
            word_logits_count = dict(Counter(word_logits))
            shortlisted_aspects = [k for k,v in word_logits_count.items() if v ==  max(word_logits_count.values())] 
            word_logit = shortlisted_aspects[0]
            if len(shortlisted_aspects) > 1:
                if word_logits[0] in shortlisted_aspects:
                    word_logit = word_logits[0]
            return word_logit
            
        for t, l in list(zip(outputs[0], list(logits))):
            t = self.tokenizer.decode(t)
            print('TRUE', t, l)
            if t not in additional_special_tokens:
            

                if t != t.strip():
                    word_logit = process_logits(word_logits)
                    
                    all_w_logits.append((''.join(w), word_logit))
                    w = [t]
                    word_logits = [l]

                else:
                    w.append(t)
                    word_logits.append(l)
            else:
                all_w_logits.append((''.join(w), process_logits(word_logits)))
                w = []
                word_logits = []
        
        if w:
            all_w_logits.append((''.join(w), process_logits(word_logits)))
        
        print(all_w_logits)
    
        #model_output = ' '.join([self.tokenizer.decode(w) for w in outputs])
        model_output = [each[0].strip() for each in all_w_logits[1:]]
        logits = [each[1] for each in all_w_logits[1:]]
        #model_output = ' '.join([w for w in model_output.split(' ') if w not in additional_special_tokens])
        
        #pred_direction = self._get_summary_direction(model_output)
        #print('summarizing with templates ...')
        #model_outputs_temp_diff, model_outputs_temp_nodiff = self.template_summary(batch)
        
        #temp_dict = {'nodiff': model_outputs_temp_nodiff, 'diff':  model_outputs_temp_diff, 'pred_direction': pred_direction}
        summary = ' '.join(model_output)
        print('SUMMARY', summary, summary.split(' '))
        print('LOGITS', [each[1] for each in all_w_logits])
        return {'summary': summary, 'aspect_indices': logits }
