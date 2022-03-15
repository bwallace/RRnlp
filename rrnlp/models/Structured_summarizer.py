import torch
from dataclasses import dataclass
from torch import nn
from transformers.models.led.modeling_led import LEDModel, shift_tokens_right, LEDPreTrainedModel
from transformers.models.led.configuration_led import LEDConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
import copy
from typing import Optional, Tuple
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    Seq2SeqLMOutput
)
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
    lm_logits_individual: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input_ind = torch.argmax(input, 2, keepdim=True)
        one_hot = torch.FloatTensor(input.shape)
        input_ind = input_ind.to(device=one_hot.device)
        one_hot.zero_()
        one_hot.scatter_(2, input_ind, 1)
        alphas = one_hot
        alphas = alphas.to(device=input.device)
        return alphas

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class LEDForDataToTextGeneration_MultiLM_SoftSupervised(LEDPreTrainedModel):
    # base_model_prefix = "model"
    # _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)

        self.register_buffer("final_logits_bias0", torch.zeros((1, self.led.shared.num_embeddings)))
        self.register_buffer("final_logits_bias1", torch.zeros((1, self.led.shared.num_embeddings)))
        self.register_buffer("final_logits_bias2", torch.zeros((1, self.led.shared.num_embeddings)))
        self.register_buffer("final_logits_bias3", torch.zeros((1, self.led.shared.num_embeddings)))
        self.register_buffer("final_logits_bias4", torch.zeros((1, self.led.shared.num_embeddings)))

        self.out_drop = nn.Dropout(p=0.2)
        self.proj = nn.Linear(config.d_model, 1)
        self.softmax_logits = nn.LogSoftmax(dim=2)
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)

        self.activation_fn = ACT2FN['relu']
        self.soft_weigh = nn.Softmax(dim=-1)
        self.init_weights()
        # self.post_init()

    def _make_multiple_lm_heads(self):
        self.lm_head1 = copy.deepcopy(self.lm_head)
        self.lm_head2 = copy.deepcopy(self.lm_head)
        self.lm_head3 = copy.deepcopy(self.lm_head)
        return

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

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
        old_num_tokens = self.final_logits_bias0.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias0, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias0", new_bias)

        old_num_tokens = self.final_logits_bias1.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias1, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias1", new_bias)

        old_num_tokens = self.final_logits_bias2.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias2, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias2", new_bias)

        old_num_tokens = self.final_logits_bias3.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias3, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias3", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.lm_head1 = new_embeddings
        self.lm_head2 = new_embeddings
        self.lm_head3 = new_embeddings

    def forward(
            self,
            input_ids_col0=None,
            input_ids_col1=None,
            input_ids_col2=None,
            input_ids_col3=None,
            attention_mask_col0=None,
            attention_mask_col1=None,
            attention_mask_col2=None,
            attention_mask_col3=None,
            global_attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs_col0=None,
            encoder_outputs_col1=None,
            encoder_outputs_col2=None,
            encoder_outputs_col3=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            decoder_time_step=None,
            cross_attn_head_mask=None,
            labels=None,
            labels_tagged=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            control_key=None,
            inference=None,
            background_lm=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if labels_tagged is not None:
            labels_tagged[labels_tagged == 4] = -100

        outputs0 = self.led(
            input_ids_col0,
            attention_mask=attention_mask_col0,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs_col0,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[0] if past_key_values else None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs1 = self.led(
            input_ids_col1,
            attention_mask=attention_mask_col1,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs_col1,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[1] if past_key_values else None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs2 = self.led(
            input_ids_col2,
            attention_mask=attention_mask_col2,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs_col2,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[2] if past_key_values else None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs3 = self.led(
            input_ids_col3,
            attention_mask=attention_mask_col3,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs_col3,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[3] if past_key_values else None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        alphas_0 = self.proj(self.out_drop(outputs0[0]))
        alphas_1 = self.proj(self.out_drop(outputs1[0]))
        alphas_2 = self.proj(self.out_drop(outputs2[0]))
        alphas_3 = self.proj(self.out_drop(outputs3[0]))

        alphas = torch.cat([alphas_0, alphas_1, alphas_2, alphas_3], dim=-1)
        alphas = self.soft_weigh(alphas)

        token_loss = None
        if labels_tagged is not None:
            bs = alphas.shape[0]
            tok_lens = alphas.shape[1]
            token_loss_fct = nn.CrossEntropyLoss()
            token_loss = token_loss_fct(alphas.view(bs * tok_lens, -1), labels_tagged.view(-1))

        if inference:
            alphas_ind = torch.argmax(alphas, 2, keepdim=True)
            one_hot = torch.FloatTensor(alphas.shape)
            alphas_ind = alphas_ind.to(device=one_hot.device)
            one_hot.zero_()
            one_hot.scatter_(2, alphas_ind, 1)
            alphas = one_hot
            alphas = alphas.to(device=outputs3[0].device)

        lm_logits0 = self.softmax_logits(self.lm_head(outputs0[0]) + self.final_logits_bias0)
        lm_logits1 = self.softmax_logits(self.lm_head1(outputs1[0]) + self.final_logits_bias1)
        lm_logits2 = self.softmax_logits(self.lm_head2(outputs2[0]) + self.final_logits_bias2)
        lm_logits3 = self.softmax_logits(self.lm_head3(outputs3[0]) + self.final_logits_bias3)

        lm_logits = [alphas[batch_id][:, 0][:, None] * lm_logits0[batch_id].unsqueeze(0) + \
                     alphas[batch_id][:, 1][:, None] * lm_logits1[batch_id].unsqueeze(0) + \
                     alphas[batch_id][:, 2][:, None] * lm_logits2[batch_id].unsqueeze(0) + \
                     alphas[batch_id][:, 3][:, None] * lm_logits3[batch_id].unsqueeze(0) \
                     for batch_id in range(0, lm_logits0.shape[0])]

        if not control_key:
            lm_logits = torch.cat(lm_logits)

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
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            if token_loss:
                masked_lm_loss = masked_lm_loss + token_loss

        if not return_dict:
            output = (lm_logits) + outputs3[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        lm_logits_list = [torch.stack([alphas[batch_id][:, 0], \
                                       alphas[batch_id][:, 1], \
                                       alphas[batch_id][:, 2], \
                                       alphas[batch_id][:, 3]]) \
                          for batch_id in range(0, lm_logits0.shape[0])]
        lm_logits_list = torch.stack(lm_logits_list)

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=[outputs0.past_key_values, outputs1.past_key_values, outputs2.past_key_values,
                             outputs3.past_key_values],
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
            lm_logits_individual=lm_logits_list
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask_col0=None,
            attention_mask_col1=None,
            attention_mask_col2=None,
            attention_mask_col3=None,
            global_attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs_col0=None,
            encoder_outputs_col1=None,
            encoder_outputs_col2=None,
            encoder_outputs_col3=None,
            control_key=None,
            inference=None,
            background_lm=None,
            **kwargs
    ):
        decoder_time_step = decoder_input_ids.shape[1] - 1
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids_col0": None,
            "input_ids_col1": None,
            "input_ids_col2": None,
            "input_ids_col3": None,
            "decoder_time_step": decoder_time_step,
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
            "inference": inference,
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


class StructuredSummaryBot:
    def __init__(self):
        self.model = None

    def summarize(self, p_spans, i_spans, o_spans, punchline_text):
        # TODO: replace with real generated summary
        # currently this is just a dummy summarizer that spits out the punchline of the first study
        return {'summary': punchline_text[0]}
