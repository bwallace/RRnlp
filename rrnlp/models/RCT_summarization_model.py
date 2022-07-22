from json import encoder
from re import L
import torch
from dataclasses import dataclass
from torch import nn
from transformers.models.led.modeling_led import shift_tokens_right, LEDPreTrainedModel, LEDEncoderBaseModelOutput, LEDEncoder, LEDDecoder
from transformers.models.led.configuration_led import LEDConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
import copy
from typing import Optional, Tuple
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    Seq2SeqLMOutput
)
from rrnlp.models.util.rct_summarize.utils import _tie_decoder_weights
import torch.nn.functional as F 


@dataclass
class LEDSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    lm_logits_individual : Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class LEDForDataToTextGeneration_MultiLM(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", \
                                        r"lm_head_pop\.weight", \
                                        r"lm_head_int\.weight", \
                                        r"lm_head_out\.weight", \
                                        r"lm_head_ptext\.weight", \
                                        r"lm_head_bg\.weight"]

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = LEDEncoder(config, self.shared)
        self.decoder = LEDDecoder(config, self.shared)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))

        
        self.proj = nn.Linear(config.d_model , 1)

        self.activation_fn = ACT2FN['relu']
        self.softmax = nn.Softmax(dim =-1)
        
        self.init_weights()


    def _make_decoders(self, num_decoder_layers_shared, background_lm = False):
        self.num_decoder_layers_shared = num_decoder_layers_shared
        self.decoder1 = copy.deepcopy(self.decoder)
        self.decoder2 = copy.deepcopy(self.decoder)
        self.decoder3 = copy.deepcopy(self.decoder)

        for k in range(num_decoder_layers_shared):
            _tie_decoder_weights(self.decoder1.layers[k],
                                self.decoder.layers[k], f'decoder_layer{k}')
            _tie_decoder_weights(self.decoder2.layers[k],
                                self.decoder.layers[k], f'decoder_layer{k}')
            _tie_decoder_weights(self.decoder3.layers[k],
                                self.decoder.layers[k], f'decoder_layer{k}')

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_func(self, final_logits_bias, new_num_tokens, old_num_tokens):
        if new_num_tokens <= old_num_tokens:
            new_bias = final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=final_logits_bias.device)
            new_bias = torch.cat([final_logits_bias, extra_bias], dim=1)
        return new_bias


    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias", new_bias)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_mixture_dist(self, outputs0, outputs1, outputs2, outputs3):
        alphas_0 = self.proj(outputs0)
        alphas_1 = self.proj(outputs1)
        alphas_2 = self.proj(outputs2)
        alphas_3 = self.proj(outputs3)

        alphas = self.softmax(torch.cat([alphas_0, 
                            alphas_1, 
                            alphas_2, 
                            alphas_3], dim = -1))
        #print(alphas)
        return alphas

    
    def forward(
        self,
        input_ids_col0 = None,
        input_ids_col1 = None,
        input_ids_col2 = None, 
        input_ids_col3 = None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        global_attention_mask = None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs_col0 = None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        decoder_time_step = None,
        cross_attn_head_mask=None,
        labels=None,
        labels_tagged = None,
        labels_tagged_weights = None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        control_key = None,
        inference = None,
        background_lm = None,
    ):


        def get_encoder_outputs(encoder_outputs, input_ids, attention_mask, global_attention_mask):
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
            elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
                encoder_outputs = LEDEncoderBaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                )
            return encoder_outputs

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Using this like Bart, as LED is derived from it. So far
        # No checkpoint on the hub exists that uses that in practice.
        # https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                

        encoder_outputs_col0 = get_encoder_outputs(encoder_outputs_col0, input_ids_col0, attention_mask_col0, global_attention_mask)
        encoder_outputs_col1 = get_encoder_outputs(encoder_outputs_col1, input_ids_col1, attention_mask_col1, global_attention_mask)
        encoder_outputs_col2 = get_encoder_outputs(encoder_outputs_col2, input_ids_col2, attention_mask_col2, global_attention_mask)
        encoder_outputs_col3 = get_encoder_outputs(encoder_outputs_col3, input_ids_col3, attention_mask_col3, global_attention_mask)

        decoder_outputs0 = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs_col0[0],
            encoder_attention_mask=attention_mask_col0,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[0] if past_key_values else None,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        decoder_outputs1 = self.decoder1(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs_col1[0],
            encoder_attention_mask=attention_mask_col1,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[1] if past_key_values else None,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        decoder_outputs2 = self.decoder2(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs_col2[0],
            encoder_attention_mask=attention_mask_col2,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[2] if past_key_values else None,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        decoder_outputs3 = self.decoder3(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs_col3[0],
            encoder_attention_mask=attention_mask_col3,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[3] if past_key_values else None,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

       
        #print('LEN', len(decoder_outputs0.hidden_states))
        alphas = self.get_mixture_dist(decoder_outputs0[0],
                                        decoder_outputs1[0],
                                        decoder_outputs2[0],
                                        decoder_outputs3[0])
        
        lm_logits0 = F.linear(decoder_outputs0[0], self.shared.weight, bias=self.final_logits_bias)
        lm_logits1 = F.linear(decoder_outputs1[0], self.shared.weight, bias=self.final_logits_bias)
        lm_logits2 = F.linear(decoder_outputs2[0], self.shared.weight, bias=self.final_logits_bias)
        lm_logits3 = F.linear(decoder_outputs3[0], self.shared.weight, bias=self.final_logits_bias)

        lm_logits0 = F.softmax(lm_logits0, -1)
        lm_logits1 = F.softmax(lm_logits1, -1)
        lm_logits2 = F.softmax(lm_logits2, -1)
        lm_logits3 = F.softmax(lm_logits3, -1)
        token_loss = None
        
        if labels_tagged is not None:
            #print(labels_tagged)
            labels_tagged[labels_tagged == 4] = -100 
            #print(labels_tagged)
            #print('===')
            bs = alphas.shape[0]
            tok_lens = alphas.shape[1]
            token_loss_fct = nn.NLLLoss()
            token_loss = token_loss_fct(torch.log(alphas.view(bs * tok_lens, -1)), labels_tagged.view(-1))
            
        #print(token_loss)
        if False:
            alphas_ind = torch.argmax(alphas, 2, keepdim=True)
            one_hot = torch.FloatTensor(alphas.shape)
            alphas_ind = alphas_ind.to(device = one_hot.device)
            one_hot.zero_()
            one_hot.scatter_(2, alphas_ind, 1)
            alphas = one_hot
            alphas = alphas.to(device = decoder_outputs0[0].device)

        
        lm_logits = torch.mul(alphas[:, :, 0].unsqueeze(2), lm_logits0) + \
                    torch.mul(alphas[:, :, 1].unsqueeze(2), lm_logits1)  + \
                    torch.mul(alphas[:, :, 2].unsqueeze(2), lm_logits2) + \
                    torch.mul(alphas[:, :, 3].unsqueeze(2), lm_logits3) 

        if not control_key: 
            lm_logits = lm_logits

        if control_key == 'population':
            lm_logits = lm_logits0

        elif control_key == 'intervention':
            lm_logits = lm_logits1

        elif control_key == 'outcome':
            lm_logits = lm_logits2

        elif control_key == 'punchline_text':
            lm_logits = lm_logits3

        
        masked_lm_loss = None

        if labels is not None:
            loss_fct = nn.NLLLoss()
            masked_lm_loss = loss_fct(torch.log(lm_logits.view(-1, self.config.vocab_size)), labels.view(-1))
            if token_loss:
                masked_lm_loss = masked_lm_loss + token_loss
               
        lm_logits_list = [torch.stack([alphas[batch_id][:,0]  , \
                                        alphas[batch_id][:,1],  \
                                        alphas[batch_id][:,2] , \
                                        alphas[batch_id][:,3]]) \
                for batch_id in range(0, lm_logits0.shape[0])]
        lm_logits_list = torch.stack(lm_logits_list)
        
        
        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=[decoder_outputs0.past_key_values, \
                            decoder_outputs1.past_key_values, \
                            decoder_outputs2.past_key_values, \
                            decoder_outputs3.past_key_values],
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
            lm_logits_individual = lm_logits_list
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        global_attention_mask = None,
        head_mask=None,
        use_cache=None,
        encoder_outputs_col0 =None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        control_key = None,
        inference= None,
        background_lm = None,
        **kwargs
    ):
        decoder_time_step =  decoder_input_ids.shape[1] - 1
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids_col0": None,
            "input_ids_col1": None,
            "input_ids_col2": None,
            "input_ids_col3": None,
            "decoder_time_step":decoder_time_step,
            "encoder_outputs_col0": encoder_outputs_col0,
            "encoder_outputs_col1": encoder_outputs_col1,
            "encoder_outputs_col2": encoder_outputs_col2,
            "encoder_outputs_col3": encoder_outputs_col3,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask_col0": attention_mask_col0,
            "attention_mask_col1": attention_mask_col1,
            "attention_mask_col2": attention_mask_col2,
            "attention_mask_col3": attention_mask_col3,
            "global_attention_mask": global_attention_mask,
            "control_key": control_key,
            "inference" : inference,
            "background_lm": background_lm,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)

     }


    @staticmethod
    def _reorder_cache(past, beam_idx):
        past_all = []
        for past_idx in past:
            reordered_past = ()
            for layer_past in past_idx:
                reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
            past_all.append(reordered_past)
        return past_all

# @dataclass
# class LEDSeq2SeqLMOutput(Seq2SeqLMOutput):
#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
#     decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
#     cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
#     encoder_last_hidden_state: Optional[torch.FloatTensor] = None
#     encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
#     lm_logits_individual : Optional[Tuple[Tuple[torch.FloatTensor]]] = None


# class LEDForDataToTextGeneration_MultiLM_Background(LEDPreTrainedModel):
#     base_model_prefix = "led"
#     _keys_to_ignore_on_load_missing = [r"final_logits_bias", \
#                                         r"lm_head_pop\.weight", \
#                                         r"lm_head_int\.weight", \
#                                         r"lm_head_out\.weight", \
#                                         r"lm_head_ptext\.weight", \
#                                         r"lm_head_bg\.weight"]

#     def __init__(self, config: LEDConfig):
#         super().__init__(config)
#         padding_idx, vocab_size = config.pad_token_id, config.vocab_size
#         self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

#         self.encoder = LEDEncoder(config, self.shared)
#         self.decoder = LEDDecoder(config, self.shared)

#         self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))

        
#         self.proj = nn.Linear(config.d_model , 1)

#         self.activation_fn = ACT2FN['relu']
#         self.softmax = nn.Softmax(dim =-1)
        
#         self.init_weights()


#     def _make_decoders(self, num_decoder_layers_shared, background_lm):
#         self.num_decoder_layers_shared = num_decoder_layers_shared
#         self.background_lm = background_lm
        
        
#         self.decoder1 = copy.deepcopy(self.decoder)
#         self.decoder2 = copy.deepcopy(self.decoder)
#         self.decoder3 = copy.deepcopy(self.decoder)
#         if self.background_lm:
#             self.decoder4 = copy.deepcopy(self.decoder)

#         for k in range(num_decoder_layers_shared):
#             _tie_decoder_weights(self.decoder.layers[k],
#                                 self.decoder1.layers[k], f'decoder_layer{k}')
#             _tie_decoder_weights(self.decoder.layers[k],
#                                 self.decoder2.layers[k], f'decoder_layer{k}')
#             _tie_decoder_weights(self.decoder.layers[k],
#                                 self.decoder3.layers[k], f'decoder_layer{k}')
#             if self.background_lm:
#                 _tie_decoder_weights(self.decoder.layers[k],
#                                     self.decoder4.layers[k], f'decoder_layer{k}')

#     def get_encoder(self):
#         return self.encoder

#     def get_decoder(self):
#         return self.decoder

#     def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         new_embeddings = super().resize_token_embeddings(new_num_tokens)
#         self._resize_final_logits_bias(new_num_tokens)
#         return new_embeddings

#     def _resize_func(self, final_logits_bias, new_num_tokens, old_num_tokens):
#         if new_num_tokens <= old_num_tokens:
#             new_bias = final_logits_bias[:, :new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=final_logits_bias.device)
#             new_bias = torch.cat([final_logits_bias, extra_bias], dim=1)
#         return new_bias


#     def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
#         old_num_tokens = self.final_logits_bias.shape[-1]
#         new_bias = self._resize_func(self.final_logits_bias, new_num_tokens, old_num_tokens)
#         self.register_buffer("final_logits_bias", new_bias)

#     def get_input_embeddings(self):
#         return self.shared

#     def set_input_embeddings(self, value):
#         self.shared = value
#         self.encoder.embed_tokens = self.shared
#         self.decoder.embed_tokens = self.shared

#     def get_mixture_dist(self, outputs0, outputs1, outputs2, outputs3, outputs4 = None):
#         alphas_0 = self.proj(outputs0)
#         alphas_1 = self.proj(outputs1)
#         alphas_2 = self.proj(outputs2)
#         alphas_3 = self.proj(outputs3)
#         if outputs4:
#             alphas_4 = self.proj(outputs4)
        
#         if outputs4:
#             alphas = self.softmax(torch.cat([alphas_0, 
#                                 alphas_1, 
#                                 alphas_2, 
#                                 alphas_3,
#                                 alphas_4], dim = -1))
#         else:
#             alphas = self.softmax(torch.cat([alphas_0, 
#                                 alphas_1, 
#                                 alphas_2, 
#                                 alphas_3], dim = -1))
#         #print(alphas)
#         return alphas

    
#     def forward(
#         self,
#         input_ids_col0 = None,
#         input_ids_col1 = None,
#         input_ids_col2 = None, 
#         input_ids_col3 = None,
#         input_ids_col4 = None,
#         attention_mask_col0 = None,
#         attention_mask_col1 = None,
#         attention_mask_col2 = None,
#         attention_mask_col3 = None,
#         attention_mask_col4 = None,
#         global_attention_mask = None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         encoder_outputs_col0 = None,
#         encoder_outputs_col1 = None,
#         encoder_outputs_col2 = None,
#         encoder_outputs_col3 = None,
#         encoder_outputs_col4 = None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         decoder_time_step = None,
#         cross_attn_head_mask=None,
#         labels=None,
#         labels_tagged = None,
#         labels_tagged_weights = None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=True,
#         return_dict=None,
#         control_key = None,
#         inference = None,
#         background_lm = None,
#     ):


#         def get_encoder_outputs(encoder_outputs, input_ids, attention_mask, global_attention_mask):
#             if encoder_outputs is None:
#                 encoder_outputs = self.encoder(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     global_attention_mask=global_attention_mask,
#                     head_mask=head_mask,
#                     inputs_embeds=inputs_embeds,
#                     output_attentions=output_attentions,
#                     output_hidden_states=output_hidden_states,
#                     return_dict=return_dict,
#                 )
#             # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
#             elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
#                 encoder_outputs = LEDEncoderBaseModelOutput(
#                     last_hidden_state=encoder_outputs[0],
#                     hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                     attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#                     global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
#                 )
#             return encoder_outputs

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )

#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # Using this like Bart, as LED is derived from it. So far
#         # No checkpoint on the hub exists that uses that in practice.
#         # https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if labels is not None:
#             use_cache = False
#             if decoder_input_ids is None:
#                 decoder_input_ids = shift_tokens_right(
#                     labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                 )
                

#         encoder_outputs_col0 = get_encoder_outputs(encoder_outputs_col0, input_ids_col0, attention_mask_col0, global_attention_mask)
#         encoder_outputs_col1 = get_encoder_outputs(encoder_outputs_col1, input_ids_col1, attention_mask_col1, global_attention_mask)
#         encoder_outputs_col2 = get_encoder_outputs(encoder_outputs_col2, input_ids_col2, attention_mask_col2, global_attention_mask)
#         encoder_outputs_col3 = get_encoder_outputs(encoder_outputs_col3, input_ids_col3, attention_mask_col3, global_attention_mask)
#         if background_lm:
#             encoder_outputs_col4 = get_encoder_outputs(encoder_outputs_col4, input_ids_col4, attention_mask_col4, None)

#         decoder_outputs0 = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs_col0[0],
#             encoder_attention_mask=attention_mask_col0,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values[0] if past_key_values else None,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         decoder_outputs1 = self.decoder1(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs_col1[0],
#             encoder_attention_mask=attention_mask_col1,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values[1] if past_key_values else None,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         decoder_outputs2 = self.decoder2(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs_col2[0],
#             encoder_attention_mask=attention_mask_col2,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values[2] if past_key_values else None,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         decoder_outputs3 = self.decoder3(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs_col3[0],
#             encoder_attention_mask=attention_mask_col3,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values[3] if past_key_values else None,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         if background_lm:
#             decoder_outputs4 = self.decoder4(
#                 input_ids=decoder_input_ids,
#                 attention_mask=decoder_attention_mask,
#                 encoder_hidden_states=encoder_outputs_col4[0],
#                 encoder_attention_mask=attention_mask_col4,
#                 head_mask=decoder_head_mask,
#                 cross_attn_head_mask=cross_attn_head_mask,
#                 past_key_values=past_key_values[4] if past_key_values else None,
#                 inputs_embeds=decoder_inputs_embeds,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#         #print('LEN', len(decoder_outputs0.hidden_states))
#         if background_lm:
#             alphas = self.get_mixture_dist(decoder_outputs0.hidden_states[self.num_decoder_layers_shared],
#                                             decoder_outputs1.hidden_states[self.num_decoder_layers_shared],
#                                             decoder_outputs2.hidden_states[self.num_decoder_layers_shared],
#                                             decoder_outputs3.hidden_states[self.num_decoder_layers_shared],
#                                             decoder_outputs4.hidden_states[self.num_decoder_layers_shared])
#         else:
#             alphas = self.get_mixture_dist(decoder_outputs0.hidden_states[self.num_decoder_layers_shared],
#                                             decoder_outputs1.hidden_states[self.num_decoder_layers_shared],
#                                             decoder_outputs2.hidden_states[self.num_decoder_layers_shared],
#                                             decoder_outputs3.hidden_states[self.num_decoder_layers_shared])
            
            
        
#         lm_logits0 = F.linear(decoder_outputs0[0], self.shared.weight, bias=self.final_logits_bias)
#         lm_logits1 = F.linear(decoder_outputs1[0], self.shared.weight, bias=self.final_logits_bias)
#         lm_logits2 = F.linear(decoder_outputs2[0], self.shared.weight, bias=self.final_logits_bias)
#         lm_logits3 = F.linear(decoder_outputs3[0], self.shared.weight, bias=self.final_logits_bias)
#         if background_lm:
#             lm_logits4 = F.linear(decoder_outputs4[0], self.shared.weight, bias=self.final_logits_bias)

#         lm_logits0 = F.softmax(lm_logits0, -1)
#         lm_logits1 = F.softmax(lm_logits1, -1)
#         lm_logits2 = F.softmax(lm_logits2, -1)
#         lm_logits3 = F.softmax(lm_logits3, -1)
        
#         if background_lm:
#             lm_logits4 = F.softmax(lm_logits4, -1)
            
#         if background_lm:
#             lm_logits = torch.mul(alphas[:, :, 0].unsqueeze(2), lm_logits0) + \
#                         torch.mul(alphas[:, :, 1].unsqueeze(2), lm_logits1)  + \
#                         torch.mul(alphas[:, :, 2].unsqueeze(2), lm_logits2) + \
#                         torch.mul(alphas[:, :, 3].unsqueeze(2), lm_logits3) + \
#                         torch.mul(alphas[:, :, 4].unsqueeze(2), lm_logits4)
#         else:
#             lm_logits = torch.mul(alphas[:, :, 0].unsqueeze(2), lm_logits0) + \
#                         torch.mul(alphas[:, :, 1].unsqueeze(2), lm_logits1)  + \
#                         torch.mul(alphas[:, :, 2].unsqueeze(2), lm_logits2) + \
#                         torch.mul(alphas[:, :, 3].unsqueeze(2), lm_logits3)

#         token_loss = None
#         if labels_tagged is not None:
#             bs = alphas.shape[0]
#             tok_lens = alphas.shape[1]
#             token_loss_fct = nn.CrossEntropyLoss()
#             token_loss = token_loss_fct(alphas.view(bs * tok_lens, -1), labels_tagged.view(-1))
            
#         #print(token_loss)
#         if inference:
#             alphas_ind = torch.argmax(alphas, 2, keepdim=True)
#             one_hot = torch.FloatTensor(alphas.shape)
#             alphas_ind = alphas_ind.to(device = one_hot.device)
#             one_hot.zero_()
#             one_hot.scatter_(2, alphas_ind, 1)
#             alphas = one_hot
#             alphas = alphas.to(device = decoder_outputs0[0].device)

        
#         if not control_key: 
#             lm_logits = lm_logits

#         if control_key == 'population':
#             lm_logits = lm_logits0

#         elif control_key == 'interventions':
#             lm_logits = lm_logits1

#         elif control_key == 'outcomes':
#             lm_logits = lm_logits2

#         elif control_key == 'punchline_text':
#             lm_logits = lm_logits3

#         lm_logits = torch.log(lm_logits)
#         masked_lm_loss = None

#         if labels is not None:
#             loss_fct = nn.NLLLoss()
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
#             if token_loss:
#                 masked_lm_loss = masked_lm_loss + token_loss
               
#         if background_lm:
#             lm_logits_list = [torch.stack([alphas[batch_id][:,0]  , \
#                                             alphas[batch_id][:,1],  \
#                                             alphas[batch_id][:,2] , \
#                                             alphas[batch_id][:,3],
#                                             alphas[batch_id][:,4]]) \
#                     for batch_id in range(0, lm_logits0.shape[0])]
#         else:
#             lm_logits_list = [torch.stack([alphas[batch_id][:,0]  , \
#                                             alphas[batch_id][:,1],  \
#                                             alphas[batch_id][:,2] , \
#                                             alphas[batch_id][:,3]]) \
#                     for batch_id in range(0, lm_logits0.shape[0])]
            
#         lm_logits_list = torch.stack(lm_logits_list)
        
        
#         past_key_values=[decoder_outputs0.past_key_values, \
#                             decoder_outputs1.past_key_values, \
#                             decoder_outputs2.past_key_values, \
#                             decoder_outputs3.past_key_values, ]
#         if background_lm:
#             past_key_values += [decoder_outputs4.past_key_values]
            
#         return LEDSeq2SeqLMOutput(
#             loss=masked_lm_loss,
#             logits=lm_logits,
#             past_key_values= past_key_values,
#             decoder_hidden_states=None,
#             decoder_attentions=None,
#             cross_attentions=None,
#             encoder_last_hidden_state=None,
#             encoder_hidden_states=None,
#             encoder_attentions=None,
#             lm_logits_individual = lm_logits_list
#         )

#     def prepare_inputs_for_generation(
#         self,
#         decoder_input_ids,
#         past=None,
#         attention_mask_col0 = None,
#         attention_mask_col1 = None,
#         attention_mask_col2 = None,
#         attention_mask_col3 = None,
#         attention_mask_col4 = None,
#         global_attention_mask = None,
#         head_mask=None,
#         use_cache=None,
#         encoder_outputs_col0 =None,
#         encoder_outputs_col1 = None,
#         encoder_outputs_col2 = None,
#         encoder_outputs_col3 = None,
#         encoder_outputs_col4 = None,
#         control_key = None,
#         inference= None,
#         background_lm = None,
#         **kwargs
#     ):
#         decoder_time_step =  decoder_input_ids.shape[1] - 1
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             decoder_input_ids = decoder_input_ids[:, -1:]

#         return {
#             "input_ids_col0": None,
#             "input_ids_col1": None,
#             "input_ids_col2": None,
#             "input_ids_col3": None,
#             "decoder_time_step":decoder_time_step,
#             "encoder_outputs_col0": encoder_outputs_col0,
#             "encoder_outputs_col1": encoder_outputs_col1,
#             "encoder_outputs_col2": encoder_outputs_col2,
#             "encoder_outputs_col3": encoder_outputs_col3,
#             "encoder_outputs_col4": encoder_outputs_col4,
#             "past_key_values": past,
#             "decoder_input_ids": decoder_input_ids,
#             "attention_mask_col0": attention_mask_col0,
#             "attention_mask_col1": attention_mask_col1,
#             "attention_mask_col2": attention_mask_col2,
#             "attention_mask_col3": attention_mask_col3,
#             "attention_mask_col4": attention_mask_col4,
#             "global_attention_mask": global_attention_mask,
#             "control_key": control_key,
#             "inference" : inference,
#             "background_lm": background_lm,
#             "head_mask": head_mask,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)

#      }


#     @staticmethod
#     def _reorder_cache(past, beam_idx):
#         past_all = []
#         for past_idx in past:
#             reordered_past = ()
#             for layer_past in past_idx:
#                 reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
#             past_all.append(reordered_past)
#         return past_all
