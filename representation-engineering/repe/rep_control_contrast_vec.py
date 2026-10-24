from transformers import BaseStreamer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation import GreedySearchDecoderOnlyOutput
from repe.rep_control_reading_vec import WrappedBlock
import torch
from torch import nn
from typing import Any, List, Optional, Tuple, Union
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from functools import partial
import warnings
import torch.distributed as dist
from logging import getLogger, logger
from functools import lru_cache
from typing import Dict
from torch.nn import CrossEntropyLoss

# === should work for Llama and Mistral ===
class ContrastGreedySearch:
    def __init__(self, 
                 attention_mask: Optional[torch.Tensor] = None,
                 eos_token_id: Optional[Union[int, List[int]]] = None,
                 inputs_embeds: Optional[torch.FloatTensor] = None,
                 input_ids: torch.LongTensor=None,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 max_length: Optional[int] = None,
                 
                 output_scores: Optional[bool] = None,
                 pad_token_id: Optional[int] = None,
                 past_key_values: Optional[List[torch.FloatTensor]] = None,
                 position_ids: Optional[torch.LongTensor] = None,
                 return_dict: Optional[bool] = None,
                 return_dict_in_generate: Optional[bool] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 streamer: Optional["BaseStreamer"] = None,
                 synced_gpus: bool = False,
                 use_cache: Optional[bool] = None,
                 model_kwargs: Optional[Dict] = None,
                
                **kwargs):
        self.input_ids = input_ids
        self.logits_processor = (logits_processor if logits_processor 
                                 is not None else LogitsProcessorList())
        self.stopping_criteria = (stopping_criteria if stopping_criteria 
                                  is not None else StoppingCriteriaList())
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.output_scores = output_scores
        self.return_dict_in_generate = return_dict_in_generate
        self.synced_gpus = synced_gpus
        self.streamer = streamer
        self.model_kwargs = model_kwargs

    # ===== pop repe/contrast control args ====
    def pop_repe_contrast_args(self)-> Tuple:
        ''' 
        Pops the repe/contrast control args from model_kwargs and return them.
        Args:
            model_kwargs: dict, the model kwargs to pop from.
        Returns:
            alpha: float, the alpha value for contrastive control.
            contrast_tokens: int, the number of contrast tokens.
            compute_contrast: bool, whether to compute contrast.
            pos_input_ids: torch.LongTensor, the positive input ids.
            pos_attention_mask: torch.LongTensor, the positive attention mask.
            neg_input_ids: torch.LongTensor, the negative input ids.
            neg_attention_mask: torch.LongTensor, the negative attention mask.
            control_layer_ids: list, the layer ids to apply control to.
        >>> Example:
            >>> pop_repe_contrast_args(model_kwargs)
            (0.5, 10, True, tensor([[1, 2, 3]]), tensor([[1, 1, 1]]), tensor([[4, 5, 6]]), tensor([[1, 1, 1]]), [0, 1])

        '''
        alpha = self.model_kwargs.pop('alpha', None)
        contrast_tokens = self.model_kwargs.pop('contrast_tokens', None)
        compute_contrast = self.model_kwargs.pop('compute_contrast', None)
        pos_input_ids = self.model_kwargs.pop('pos_input_ids', None)
        pos_attention_mask = self.model_kwargs.pop('pos_attention_mask', None)
        neg_input_ids = self.model_kwargs.pop('neg_input_ids', None)
        neg_attention_mask = self.model_kwargs.pop('neg_attention_mask', None)
        control_layer_ids = self.model_kwargs.pop('control_layer_ids', None)

        return alpha, contrast_tokens, compute_contrast, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, control_layer_ids

           
    def assert_repe_contrast_args(self,)-> None:
        '''
        Asserts that the necessary arguments for contrastive control are present in model_kwargs if compute_contrast is True.
        Args:
            compute_contrast: bool, whether to compute contrast.
            model_kwargs: dict, the model kwargs to check.
        Raises:
            AssertionError: if any of the necessary arguments are missing in model_kwargs.
        >>> Example:
            >>> assert_repe_contrast_args(model_kwargs)
            AssertionError: Please provide alpha for contrastive control
            '''
        assert 'alpha' in self.model_kwargs, "Please provide alpha for contrastive control"
        assert 'contrast_tokens' in self.model_kwargs, "Please provide contrast_tokens for contrastive control"
        assert 'compute_contrast' in self.model_kwargs, "Please provide compute_contrast for contrastive control"
        assert 'pos_input_ids' in self.model_kwargs, "Please provide pos_input_ids for contrastive control"
        assert 'pos_attention_mask' in self.model_kwargs, "Please provide pos_attention_mask for contrastive control"
        assert 'neg_input_ids' in self.model_kwargs, "Please provide neg_input_ids for contrastive control"
        assert 'neg_attention_mask' in self.model_kwargs, "Please provide neg_attention_mask for contrastive control"
        assert 'control_layer_ids' in self.model_kwargs, "Please provide control_layer_ids for contrastive control"
        assert not self.model_kwargs.get('compute_contrast') or not self.model_kwargs.get('use_cache', False), "Contrast Greedy Search not yet support generate with use_cache, please set model.generate(**kwargs, use_cache=False)"

        
    def init_values(self):
        '''
        Initializes the values for contrastive control by popping the necessary arguments from model_kwargs and asserting their presence.
        Args:
            model_kwargs: dict, the model kwargs to pop from and assert.
        Returns:
            alpha: float, the alpha value for contrastive control.
            contrast_tokens: int, the number of contrast tokens.
            compute_contrast: bool, whether to compute contrast.
            pos_input_ids: torch.LongTensor, the positive input ids.
            pos_attention_mask: torch.LongTensor, the positive attention mask.
            neg_input_ids: torch.LongTensor, the negative input ids.
            neg_attention_mask: torch.LongTensor, the negative attention mask.
            control_layer_ids: list, the layer ids to apply control to.
        >>> Example:
            >>> init_values(model_kwargs)
            (0.5, 10, True, tensor([[1, 2, 3]]), tensor([[1, 1, 1]]), tensor([[4, 5, 6]]), tensor([[1, 1, 1]]), [0, 1])
        '''
        logits_processor = (self.logits_processor 
                                if self.logits_processor is not None 
                                else LogitsProcessorList())
        stopping_criteria = (self.stopping_criteria 
                                 if self.stopping_criteria is not None 
                                 else StoppingCriteriaList())
        if self.max_length is not None:
            warnings.warn(
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
        stopping_criteria = validate_stopping_criteria(self.stopping_criteria, self.max_length)
        pad_token_id = (self.pad_token_id 
                            if self.pad_token_id is not None 
                            else self.generation_config.pad_token_id)
        eos_token_id = (self.eos_token_id 
                            if self.eos_token_id is not None 
                            else self.generation_config.eos_token_id)
        if isinstance(self.eos_token_id, int):
            eos_token_id = [self.eos_token_id]
        eos_token_id_tensor = (torch.tensor(self.eos_token_id).to(self.input_ids.device) 
                                   if self.eos_token_id is not None else None)
        output_scores = (self.output_scores 
                             if self.output_scores is not None 
                             else self.generation_config.output_scores)
        output_attentions = (
                self.output_attentions 
                if self.output_attentions is not None 
                else self.generation_config.output_attentions
            )
        output_hidden_states = (
                self.output_hidden_states 
                if self.output_hidden_states is not None 
                else self.generation_config.output_hidden_states
            )
        return_dict_in_generate = (
                self.return_dict_in_generate
                if self.return_dict_in_generate is not None
                else self.generation_config.return_dict_in_generate
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        eos_token_id = self.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.generation_config.eos_token_id
        eos_token_id_tensor = (
            torch.as_tensor(eos_token_id, device=self.input_ids.device)
            if eos_token_id is not None
            else None
        )

        return (logits_processor, stopping_criteria, pad_token_id, 
                eos_token_id_tensor, output_scores, output_attentions, 
                output_hidden_states, return_dict_in_generate,
                scores, decoder_attentions, cross_attentions, decoder_hidden_states
                )

        # keep track of which sequences are already finished
    def track_finished_sequences(self
                                 ) -> Tuple[torch.LongTensor, bool]:
        '''
        Tracks which sequences are already finished based on the eos_token_id and updates the unfinished_sequences tensor accordingly.
        Args:
            input_ids: torch.LongTensor, the input ids of the sequences.
            eos_token_id_tensor: torch.Tensor, the tensor containing the eos token ids.
        Returns:
            unfinished_sequences: torch.LongTensor, the tensor indicating which sequences are not yet finished.
            this_peer_finished: bool, whether the current peer has finished generating its sequence.
        >>> Example:
            >>> track_finished_sequences(input_ids, eos_token_id_tensor)
            (tensor([[1, 1, 0]]), False)
        '''
    
        unfinished_sequences = torch.ones(self.input_ids.shape[0], dtype=torch.long, device=self.input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if self.synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(self.input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
        return unfinished_sequences, this_peer_finished

    def prepare_model_inputs(self) -> Dict[str, torch.Tensor]:
        '''
        Prepares the model inputs for generation by calling the prepare_inputs_for_generation method of the model.
        Args:
            input_ids: torch.LongTensor, the input ids of the sequences.
            model_kwargs: dict, the model kwargs to pass to the prepare_inputs_for_generation method.
        Returns:
            model_inputs: dict, the prepared model inputs for generation.
        >>> Example:
            >>> prepare_model_inputs(input_ids, model_kwargs)
            {'input_ids': tensor([[1, 2, 3]]), 'attention_mask': tensor([[1, 1, 1]])}
        '''
        model_inputs = self.prepare_inputs_for_generation(self.input_ids, **self.model_kwargs)
        
        return model_inputs

    def forward_pass(self, model_inputs: Dict[str, torch.Tensor], 
                     output_attentions: bool, 
                     output_hidden_states: bool, 
                     pos_input_ids: torch.LongTensor, 
                     pos_attention_mask: torch.LongTensor, 
                     neg_input_ids: torch.LongTensor, 
                     neg_attention_mask: torch.LongTensor, 
                     contrast_tokens: int, 
                     compute_contrast: bool, 
                     alpha: float, 
                     control_layer_ids: list, 
                     this_peer_finished: bool,
                     pad_right: int = 0,) -> Tuple[dict, torch.Tensor]:
        '''
        Performs a forward pass through the model to get the next token logits.
        Args:
            model_inputs: dict, the prepared model inputs for generation.
            output_attentions: bool, whether to output attentions.
            output_hidden_states: bool, whether to output hidden states.
            pos_input_ids: torch.LongTensor, the positive input ids.
            pos_attention_mask: torch.LongTensor, the positive attention mask.
            neg_input_ids: torch.LongTensor, the negative input ids.
            neg_attention_mask: torch.LongTensor, the negative attention mask.
            contrast_tokens: int, the number of contrast tokens.
            compute_contrast: bool, whether to compute contrast.
            alpha: float, the alpha value for contrastive control.
            control_layer_ids: list, the layer ids to apply control to.
            this_peer_finished: bool, whether the current peer has finished generation.
        Returns:
            outputs: dict, the outputs of the model forward pass containing the next token logits and other relevant information.
            next_token_logits: torch.Tensor, the logits for the next token.
        >>> Example:
            >>> forward_pass(model_inputs, output_attentions, output_hidden_states, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, contrast_tokens, compute_contrast, alpha, control_layer_ids)
            {'logits': tensor([[0.1, 0.2, 0.3]]), 'hidden_states': tensor([[0.4, 0.5, 0.6]]), 'attentions': tensor([[0.7, 0.8, 0.9]])}
        '''
        outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                pos_input_ids=pos_input_ids,
                pos_attention_mask=pos_attention_mask,
                neg_input_ids=neg_input_ids,
                neg_attention_mask=neg_attention_mask,
                contrast_tokens=contrast_tokens,
                compute_contrast=compute_contrast,
                alpha=alpha,
                control_layer_ids=control_layer_ids,
        )

        while self.synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        return outputs, next_token_logits

    def pre_process_logits(self, next_token_logits: torch.Tensor, 
                           outputs: torch.Tensor) -> torch.Tensor:
        '''
        Pre-processes the next token logits using the logits processor to apply any necessary transformations or constraints.
        Args:
            input_ids: torch.LongTensor, the input ids of the sequences.
            next_token_logits: torch.Tensor, the logits for the next token.
        Returns:
            next_tokens_scores: torch.Tensor, the processed logits for the next token after applying the logits processor.
        >>> Example:
            >>> pre_process_logits(input_ids, next_token_logits)
            tensor([[0.1, 0.2, 0.3]])
        '''

        # pre-process distribution
        next_tokens_scores = self.logits_processor(self.input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
        if self.return_dict_in_generate:
            if self.output_scores:
                scores += (next_tokens_scores,)
            if self.output_attentions:
                decoder_attentions += (
                        (outputs.attentions,)
                    )

            if self.output_hidden_states:
                decoder_hidden_states += (
                        (outputs.hidden_states,)
                    )
            return next_tokens_scores

        
    def select_next_tokens(self, next_tokens_scores: torch.Tensor,
                           unfinished_sequences: torch.LongTensor
                           ) -> torch.LongTensor:
        '''
        Selects the next tokens based on the processed logits scores by taking the argmax and handling finished sequences.
        Args:
            next_tokens_scores: torch.Tensor, the processed logits for the next token after applying the logits processor.
            unfinished_sequences: torch.LongTensor, the tensor indicating which sequences are not yet finished.
            outputs: torch.Tensor, the model outputs.
        Returns:
            next_tokens: torch.LongTensor, the selected next tokens for each sequence.
        >>> Example:
            >>> select_next_tokens(next_tokens_scores, unfinished_sequences)
            tensor([[1, 2, 3]])
        '''
        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if self.eos_token_id is not None:
            if self.pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = (next_tokens * unfinished_sequences 
                           + self.pad_token_id * (1 - unfinished_sequences))

        return next_tokens

    def update_model_inputs(self, 
                            decoder_attentions: torch.Tensor,
                            decoder_hidden_states: torch.Tensor,
                            input_ids: torch.LongTensor,
                            next_tokens: torch.LongTensor,
                            outputs: dict,
                            scores: torch.Tensor) -> Tuple[torch.LongTensor, dict]:
        '''
        Updates the model inputs for the next generation step by concatenating the selected next tokens to the input_ids and updating the model_kwargs.
        Args:
            input_ids: torch.LongTensor, the input ids of the sequences.
            next_tokens: torch.LongTensor, the selected next tokens for each sequence.
            outputs: dict, the outputs of the model forward pass containing the next token logits and other relevant information.
            scores: torch.Tensor, the scores for the next tokens.
        Returns:
            input_ids: torch.LongTensor, the updated input ids of the sequences for the next generation step.
            model_kwargs: dict, the updated model keyword arguments.
        >>> Example:
            >>> update_model_inputs(input_ids, next_tokens, outputs, scores)
            (tensor([[1, 2, 3, 4]]), {...})
        '''
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if self.streamer is not None:
            self.streamer.put(next_tokens.cpu())
            
        model_kwargs = self._update_model_kwargs_for_generation(
                outputs, self.model_kwargs
            )
        return input_ids, model_kwargs
    def set_sentence_finished(self, 
                              input_ids: torch.LongTensor,
                              next_tokens: torch.LongTensor,
                              scores: torch.Tensor, 
                              unfinished_sequences: torch.LongTensor,
                              this_peer_finished: bool
                          ) -> Tuple[torch.LongTensor, bool]:
        '''
        Sets the sentence as finished if the eos_token_id is found in the next tokens and updates the unfinished_sequences tensor accordingly.
        Args:
            next_tokens: torch.LongTensor, the selected next tokens for each sequence.
            scores: torch.Tensor, the scores for the next tokens.
            unfinished_sequences: torch.LongTensor, the tensor indicating which sequences are not yet finished.
            this_peer_finished: bool, whether the current peer has finished generation.
        Returns:
            unfinished_sequences: torch.LongTensor, the updated tensor indicating which sequences are not yet finished.
            this_peer_finished: bool, whether the current peer has finished generation.
        >>> Example:
            >>> set_sentence_finished(next_tokens, scores, unfinished_sequences, this_peer_finished)
            (tensor([[1, 1, 0]]), True)
        '''

        # if eos_token was found in one sentence, set sentence to finished
        if self.eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(self.eos_token_id_tensor.shape[0], 1).ne(self.eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                    this_peer_finished = True

        # stop if we exceed the maximum length
        if self.stopping_criteria(input_ids, scores):
            this_peer_finished = True

            while this_peer_finished and not self.synced_gpus:
                break
        return this_peer_finished
    
    def end_streamer(self) -> BaseStreamer:
        '''
        Ends the streamer if it is not None.
        Args:
            streamer: BaseStreamer, the streamer to end.
        Returns:
            streamer: BaseStreamer, the ended streamer.
        >>> Example:
            >>> end_streamer(streamer)
            <BaseStreamer object at 0x7f8c8c8c8c8c>
        '''
        if self.streamer is not None:
            self.streamer.end()
        return self.streamer

    def return_output(self,
                     input_ids: torch.LongTensor,
                     scores: torch.Tensor,
                     decoder_attentions: Optional[Tuple[torch.Tensor]] = None,
                     decoder_hidden_states: Optional[Tuple[torch.Tensor]] = None,
                     model_kwargs: Dict[str, Any] = None) -> Union[GreedySearchDecoderOnlyOutput, torch.LongTensor]:
        '''
        Returns the output of the generation process based on the return_dict_in_generate flag.
        Args:
            input_ids: torch.LongTensor, the input ids of the sequences.
            scores: torch.Tensor, the scores for the next tokens.
            decoder_attentions: Optional[Tuple[torch.Tensor]], the decoder attentions if output_attentions is True.
            decoder_hidden_states: Optional[Tuple[torch.Tensor]], the decoder hidden states if output_hidden_states is True.
            model_kwargs: Dict[str, Any], the model keyword arguments.
        Returns:
            GreedySearchDecoderOnlyOutput or torch.LongTensor, the output of the generation process based on the return_dict_in_generate flag.
        >>> Example:
            >>> return_output(input_ids, scores, decoder_attentions, decoder_hidden_states, model_kwargs)
            GreedySearchDecoderOnlyOutput(sequences=tensor([[1, 2, 3]]), scores=tensor([[0.1, 0.2, 0.3]]), attentions=(tensor([[0.4, 0.5, 0.6]]),), hidden_states=(tensor([[0.7, 0.8, 0.9]]),), past_key_values=model_kwargs.get("past_key_values"))
        '''
        if self.return_dict_in_generate:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids


    def retrieve_input_ids_and_input_embeds(self,
                                            
                                            )-> Tuple[Optional[torch.FloatTensor]]:
        '''
        Retrieves input ids and embeddings.
        Args:
            input_ids: torch.LongTensor, the input ids of the sequences.
        Returns:
            Tuple[Optional[torch.FloatTensor]], input embeddings, batch size and sequence length
        >>> Example: 
            >>> retrieve_input_ids_and_input_embeds()
            (tensor([[0.1, 0.2, 0.3]]), 1, 3)
            

            
        '''
        # retrieve input_ids and inputs_embeds
        if self.input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif self.input_ids is not None: 
            batch_size, seq_length = self.input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(self.input_ids)
        return inputs_embeds, batch_size, seq_length

    def find_past_key_values_length(self, output_attentions: Optional[bool], past_key_values: Optional[List[torch.FloatTensor]], seq_length: int) -> int:
        '''Find the length of past key values for attention mechanisms.
        
        Args:
            output_attentions (Optional[bool]): Whether to output attention weights.
            past_key_values (Optional[List[torch.FloatTensor]]): Past key values for attention.
            seq_length (int): Sequence length of the current input.

        Returns:
            int: Length of past key values.
        >>> Example:
            >>> find_past_key_values_length(output_attentions=True, past_key_values=None, seq_length=10)
            0    
        '''
        past_key_values_length = 0
        if self.use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if ((isinstance(self, LlamaModel) 
            and self._use_flash_attention_2) 
            or (isinstance(self, MistralModel) 
                and self.config._attn_implementation == "flash_attention_2")):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif (((isinstance(self, LlamaModel) 
            and self._use_sdpa) 
            or (isinstance(self, MistralModel) 
                and self.config._attn_implementation == "sdpa")) 
            and not output_attentions):
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )


        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        activations = None
        if compute_contrast:
                # ======== Compute repe =========    
                embeds_p = self.embed_tokens(pos_input_ids)
                embeds_n = self.embed_tokens(neg_input_ids)
                hidden_states_p, hidden_states_n = embeds_p, embeds_n

                pos_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    pos_attention_mask, embeds_p.shape[:2], embeds_p, past_key_values_length
                )

                neg_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    neg_attention_mask, embeds_n.shape[:2], embeds_n, past_key_values_length
                )

            for layer_id, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    # If gradient checkpointing is enabled during training
                    forward_function = partial(
                        self._gradient_checkpointing_func,
                        decoder_layer.__call__
                    )
                else:
                    # Otherwise, use the decoder layer directly
                    forward_function = decoder_layer

                # When you need to call forward_function, provide the necessary arguments
                layer_outputs = forward_function(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache
                )

                hidden_states = layer_outputs[0]
                
                # 1. if layer in target layer, we recompute activations and add
                # 2. else will add previous computed activations
                if compute_contrast:
                    # ======== Compute activations for target layers =========    
                    with torch.no_grad():
                        hidden_states_p = forward_function(
                            hidden_states_p,
                            attention_mask=pos_attention_mask,
                            use_cache=use_cache
                        )[0].detach()

                        hidden_states_n = forward_function(
                            hidden_states_n,
                            attention_mask=neg_attention_mask,
                            use_cache=use_cache
                        )[0].detach()
                    if layer_id in control_layer_ids:
                        activations = (alpha * (hidden_states_p[:, contrast_tokens:] 
                                                - hidden_states_n[:, contrast_tokens:]))
                        c_length = activations.shape[1]

                        hidden_states[:, -c_length:, :] += activations
                        hidden_states_p[:, -c_length:, :] += activations
                        hidden_states_n[:, -c_length:, :] += activations
                            
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = None
            if use_cache:
                next_cache = (next_decoder_cache.to_legacy_cache() 
                            if use_legacy_cache else next_decoder_cache)
            if not return_dict:
                return (tuple(v for v  
                            in [hidden_states, next_cache, 
                                all_hidden_states, all_self_attns] 
                                if v is not None))
            
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

def _always_true(self, *args, **kwargs):
    return True
    
class ContrastVecMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        

        # override current forward and generation functions
        self.model.forward = partial(forward_contrast_vector, self.model)
        self.greedy_search = partial(contrast_greedy_search, self)
        # turn off _validate_model_kwargs, TODO: implement _validate_model_kwargs
        self._validate_model_kwargs = partial(_always_true, self)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ===== repe ====
        **repe_kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **repe_kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ContrastVecLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # override current forward and generation functions
        self.model.forward = partial(forward_contrast_vector, self.model)
        self.greedy_search = partial(contrast_greedy_search, self)
        # turn off _validate_model_kwargs, TODO: implement _validate_model_kwargs
        self._validate_model_kwargs = partial(_always_true, self)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        # ==== repe ====
        **repe_kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **repe_kwargs,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.contrast_tokens(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ==== Helper functions ====

