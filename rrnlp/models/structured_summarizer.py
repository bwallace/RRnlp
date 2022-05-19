
from transformers import LEDTokenizer
from rrnlp.models.RCT_summarization_model import LEDForDataToTextGeneration_MultiLM_Background
from rrnlp.models.util.rct_summarize.model_inference import Data2TextGenerator
from rrnlp.models.util.rct_summarize.model_template_inference import TemplateGenerator
import pytorch_lightning as pl
import torch
import rrnlp
from rrnlp.models import ev_inf_classifier

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

        pretrained_weights = torch.load(model_weight_file)
        pretrained_weights = load_layers(pretrained_weights, self.model)
        self.model.load_state_dict(pretrained_weights)

        self.generator = Data2TextGenerator(self.model, self.tokenizer)
        self.template_generator = TemplateGenerator(self.tokenizer, torch.device(device), self.generator)
        self.templates = self.template_generator.templates
        self.ev_bot = ev_inf_classifier.EvInfBot()
        
        

    def template_summary(self, batch):

        def get_temp_outputs(templates):
            templates_generated_text = []
            for template in templates[:1]:
                generated_text = self.template_generator.get_summary(batch, template , background_lm = True,
                                    num_beams = 4, max_length = 400, min_length = 13, 
                                    repetition_penalty=1.0, length_penalty=1.0,
                                    return_dict_in_generate=False, device = device,
                                    temperature = 0.9, do_sample = False, debug = False)
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
        outputs, logits = self.generator.generate(batch, num_beams = 4,  max_length = 400, min_length = 13, \
            repetition_penalty = 1.0, length_penalty = 1.0, early_stopping = True, \
                return_dict_in_generate = False, control_key = None, no_repeat_ngram_size = 3, \
                    background_lm = True, device = torch.device(device))
        
        logits = self._get_logit_mapped(logits)

        model_output = ' '.join([self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in outputs])
        pred_direction = self._get_summary_direction(model_output)

        model_outputs_temp_diff, model_outputs_temp_nodiff = self.template_summary(batch)
        
        temp_dict = {'nodiff': model_outputs_temp_nodiff, 'diff':  model_outputs_temp_diff, 'pred_direction': pred_direction}

        return {'summary': model_output, 'aspect_indices': logits, 'template_summaries': temp_dict }