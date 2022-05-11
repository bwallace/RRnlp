from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
import pytorch_lightning as pl
import torch
from torch import nn 
import rrnlp

device = 'cuda'

def preprocess_aggregator(inp):
    def preprocess_inp(inp):
        invalid = ['<punchline_text>', '<sep>', '</study>', '<study>', '<sep>']
        inp = ' '.join([w for w in inp.split(' ') if not([ s for s in invalid if s in w]) ])
        return inp.strip()

    inp_map = {'difference': [], 'no_difference': []}
    label_map = {'― no diff': 'no_difference'}
    for each in inp.split(' </punchline_text> ')[:-1]:
        direction = each.split('<sep>')[1].strip()
        if direction.strip():
            direction = label_map[direction] if direction in label_map else 'difference'
            inp_map[direction].append(preprocess_inp(each))
    inp = "<difference> " + " <sep> ".join(inp_map['difference']) + " </difference> "
    inp += "<no_difference> " + " <sep> ".join(inp_map['no_difference']) + " </no_difference> "
    inp.strip()
    return inp

class LitEVModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, max_len, labels_tag_weights):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.labels_tag_weights = labels_tag_weights

class EVPredictor():

    def __init__(self):
        checkpoint_file = 'checkpoint_files/ev_classifier_binary_3e-5_freetop2_agg/epoch=5-val_loss=1.02.ckpt'
        path_to_data = '/scratch/ramprasad.sa/'
        tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

        additional_special_tokens = ["<sep>" , "<difference>", "</difference>", "<no_difference>", "</no_difference>"]
        tokenizer.add_tokens(additional_special_tokens)

        ev_model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096",\
             num_labels = 2, problem_type = 'single_label_classification')

        max_len = 4096
        lit_ev_model = LitEVModel.load_from_checkpoint(checkpoint_path=path_to_data + checkpoint_file, 
                                            learning_rate = 3e-5, 
                                            tokenizer = tokenizer, 
                                            model = ev_model, 
                                            max_len = max_len, 
                                            labels_tag_weights = None )   
        self.model = lit_ev_model.model
        self.model.eval()
        
        self.model.to(torch.device(device))
        self.tokenizer = tokenizer


    def model_inference(self, input_ids, decode_tokenizer):
        def run_tokenizer(snippet, tokenizer):
    
            encoded_dict = tokenizer(
                snippet,
                max_length=4096,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                )
            return encoded_dict

        inference_text = ' '.join([decode_tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in input_ids])
        #print(inference_text)
        inference_text = preprocess_aggregator(inference_text)
        print(inference_text)
        encoding = run_tokenizer(inference_text, self.tokenizer)
        input_ids = encoding['input_ids'].to(torch.device(device))
        attention_mask = encoding['attention_mask'].to(torch.device(device))
        output = self.model(input_ids, attention_mask = attention_mask)
        logits = output['logits']
        softmax = nn.Softmax(dim = -1)
        logits = softmax(logits)
        prediction = torch.argmax(logits, axis=1)
        return logits, prediction.flatten().tolist()[0]
    
