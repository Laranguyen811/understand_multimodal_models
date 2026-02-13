import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import datasets
import einops
from huggingface_hub import hf_hub_download
import re
from functools import lru_cache
import transformers
import json
from transformers import AutoTokenizer
from typing import Any, Optional, Tuple, Union, List
from torch import Tensor
from vit_prisma.prisma_tools.factored_matrix import FactoredMatrix
CACHE_DIR = transformers.TRANSFORMERS_CACHE

def download_file_from_hf(repo_name, file_name, subfolder=".",cache_dir=CACHE_DIR, force_is_torch=False):
    '''
    Helper function to download files from the HuggingFace Hub, from subfolder/file_name in repo_name, saving locally to cache_dir and returning the loaded file (if a json or torch object) and the file path otherwise.
    If it is a torch file without the ".pth" extension, set force_is_torch=True to load it as a torch object. 
    '''
    file_path = hf_hub_download(repo_id=repo_name,
                                filename=file_name,
                                subfolder=subfolder,
                                cache_dir=cache_dir)
    
    print(f"Saved at file path: {file_path}")
    if file_path.endswith(".pth") or force_is_torch: # If the file ending is .pth or force is torch
        return torch.load(file_path) # Return file path
    
    elif file_path.endswith(".json"): # If the file path ending is .json
        return json.load(open(file_path,"r")) # Open the file in read mode
    else:
        print("File type not supported:",file_path.split('.')[-1]) # Return the message and split the file path by '.'(remove any '.') and obtain the last string
        return file_path
    
def get_sample_from_dataset(sequences, nb_sample=2, print_len=10):
    '''
    Get a sample from datasets.
    '''
    rd_idx = np.random.randint(0, len(sequences), 3)
    return "\n".join([str(sequences[k][:print_len] + "..." for k in rd_idx)]) # Return a string of sequences with no whitespaces at k index and print_len

def print_gpu_mem(step_name=""):
    '''
    Prints GPU memory.
    '''
    print(
        f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30,2)} GiB allocated on GPU."
    )

def get_corner(tensor,n=3):
    '''
    Gets the top left corner of the tensor.
    '''
    return tensor[tuple(slice(n) for _ in range(tensor.ndim))] # Return the top left corner of the tensor 

def to_numpy(tensor, flat=False):
    '''
    Turns tensor into numpy.
    '''
    if (type(tensor) != torch.Tensor) and (type(tensor) != torch.nn.parameter.Parameter): # If type of tensor is different from torch.Tensor and torch.nn.parameter.Parameter
        return tensor
    if flat:
        tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()
    

def get_cross_entropy_loss(logits: torch.Tensor, tokens:torch.Tensor, return_per_token: bool = False) -> Tensor|float:

    '''
    Cross entropy loss for the model, gives the loss for predicting the NEXT token. 
    Args:
        logits (torch.Tensor): Logits. Shape[batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        return_per_token (bool, optional): A boolean of whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs).
    Returns:
        Tensor of log probabilities or float of mean of log probabilities
    '''
    log_probs = F.log_softmax(logits, dim=-1) # Get the log softmax of the logits along the last dimension

    # User torch.gather to get the log probs of the correct token at each position
    predicted_log_probs = log_probs[...,:,:].gather(dim=-1,index=tokens[...,:,None] )[...,0] # Tensor used in gather must have the same rank (input and index must have the same number of dimensions)
    if return_per_token:
        return predicted_log_probs
    else:
        return -predicted_log_probs.mean() # Return the mean of the negative predicted log probabilities
    
def model_accuracy(logits: torch.Tensor, tokens:torch.Tensor, return_per_token: bool = False
                   ) -> Tensor|float:
    '''
    Computes the accuracy of the model in predicting the next token using cross entropy.
    Args:
        logits (torch.Tensor): A tensor of logits.
        tokens (torch.Tensor[int64]): A tensor of input tokens.
        return_per_token (bool, optional): A boolean of whether to return the accuracy per token, or the mean accuracy.
    Returns:
        Float of accuracy or tensor of accuracy per token.
    '''
    top_prediction = logits.argmax(dim=-1) # Get the top prediction by taking the argmax along the last dimension
    correct = top_prediction[:,:] == tokens[:,1:] 
    if return_per_token:
       return correct # Return the correct predictions
    else: 
        return correct.sum()/correct.numel() # Return the mean accuracy

def calculate_gelu(input):
    '''
    Implementation of the gelu activation function.
    '''
    return 0.5 * input * (1 + torch.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * input ** 3)))


def keep_single_column(
        dataset: datasets.arrow_dataset.Dataset,
        column_name: str,
):
    '''
    Acts on HuggingFace datasets to keep only a single column name - useful for when we tokenize and mix together different strings.
    '''
    for key in dataset.features: # Loop through the keys in the dataset features
        if key != column_name: # If the key is not equal to the column name
            dataset = dataset.remove_columns(key) # Remove the column with the key

    return dataset

def tokenise_and_concatenate(
        dataset: Any,
        model_name_or_path: str,
        tokeniser: AutoTokenizer,
        column_name: str,
        streaming: bool = False,
        max_length: int = 1024,
        cadd_bos_token: bool = True,
        num_proc: int = 10,
    ):
    '''
    Helper function to tokenise and concatenaate a dataset. This converts the data to tokens, concatenates them (separated by EOS (End Of Sequence) tokens, and reshapes them into a a 2D array of shape (___, sequence_length), dropping the last batch. 
    Splits the string into 20, feed it into the tokeniser, in parallel with padding, then remove padding at the end.
    Args:
        dataset (any): The dataset to tokenise and concatenate.
        tokenizer (AutoTokenizer): The tokenizer to use.
        column_name (str): A string of column name to use.
        streaming (bool, optional): A boolean of whether the dataset is streaming. Defaults to False.
        max_length (int, optional): An integer of the maximum length of the sequences. Defaults to 1024.
        cadd_bos_token (bool, optional): A bolean of Whether to add a BOS (Beginning Of Sequence) token. Defaults to True.
        num_proc (int, optional): An integer of the number of processes to use. Defaults to 10.
    Returns:
        The tokenised and concatenated dataset.
    '''
    tokeniser = tokeniser.from_pretrained(pretrained_model_name_or_path=model_name_or_path) # Load the tokenizer from the pretrained model
    dataset = keep_single_column(dataset, column_name) # Keep only the column name
    if tokeniser.pad_token is None: # If the tokenizer pad token is None
        tokeniser.add_special_tokens({'pad_token': '<PAD>'}) # Add a pad token purely to implement the tokeniser. This will be removed before inputting tokens to the model.
    # Define the length to split the strings into
    if add_bos_token:
        length = max_length - 1 # If adding BOS token, reduce length by 1
    else:
        length = max_length

    def tokenise_function(examples):
        # Tokenise the examples with padding and truncation
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by EOS tokens
        full_text = tokeniser.eos_token.join(text)  
        # Divde into 20 chunks of apprixmately equal length
        num_chunks = 20
        chunk_len = (len(full_text)-1) // num_chunks + 1
        chunks = [full_text[i*chunk_len:(i+1)*chunk_len] for i in range(num_chunks)]
        # Tokenise the chunks in parallel. Use NumPy because HuggingFace map does not want tensors returned
        tokens = tokeniser(chunks, return_tensors='np', padding=True)['input_ids'].flatten()
        # Remove padding tokens
        tokens = tokens[tokens != tokeniser.pad_token_id] # Remove padding tokens
        num_tokens = len(tokens) # Get the number of tokens
        num_batches = num_tokens // length # Get the number of batches
        # Drop the final tokens if not enough to make a full sequence. Why?
        tokens = tokens[:length * num_batches]    
        tokens = einops.rearrange(tokens, '(b l) -> b l', b = num_batches, l=length) # Reshape into batches of length
        if add_bos_token:
            # Add BOS token at the start of each sequence
            prefix = np.full((num_batches, 1), tokeniser.bos_token_id) # Create a full array of prefix BOS tokens
            tokens = np.concatenate([prefix, tokens], axis=1) # Concatenate the prefix BOS tokens with the tokens along column axis
        return {'tokens': tokens}
    tokenised_dataset = dataset.map(tokenise_function, batched=True, num_proc=(num_proc if not streaming else None), remove_columns=[column_name]) # Map the tokenise function
    tokenised_dataset.set_format(type='torch', columns=['tokens']) # Set the format to torch tensors
    return tokenised_dataset

def set_seed_everywhere(seed: int)-> None:
    '''
    Sets the seed everywhere for reproducibility.
    Args:
        seed (int): The seed to set.
    Returns:
        None
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)

def sample_logits(
        final_logits: Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        temperature: float = 1.0,
        frequency_penalty: float = 0.0,
        tokens: Optional[Tensor] = None,
) -> Tensor:
    '''
    Samples logits using top-k, top-p, temperature, and frequency penalty.
    Args:
        final_logits (Tensor): The final logits to sample from. Shape [batch, vocab_size]
        top_k (Optional[int], optional): The top-k value. Defaults to None.
        top_p (Optional[int], optional): The top-p value. Defaults to None.
        temperature (float, optional): The temperature value. Defaults to 1.0.
        frequency_penalty (float, optional): The frequency penalty value. Defaults to 0.
    Returns:
        Tensor: The sampled logits. Shape [batch]
    '''
    if temperature == 0.0: # If temperature is 0, return the argmax
        # Greedy sampling
        return final_logits.argmax(dim=-1)
    else:
        # Sample from the distribution
        final_logits = final_logits / temperature
        if frequency_penalty > 0:
            assert tokens is not None, "Tokens must be provided for frequency penalty" # Assert that tokens is not None
            for batch_idx in range(final_logits.shape[0]): # Loop through the batch
                final_logits[batch_idx] -= frequency_penalty * torch.bincount(tokens[batch_idx], minlength=final_logits.shape[-1]) # Apply frequency penalty
        if top_k is not None:
            assert top_k > 0, "top_k must be greater than 0" # Assert that top_k is greater than 0
            top_logits, top_idx = final_logits.topk(top_k, dim=-1) # Get the top-k logits and indices
            indices_to_remove = final_logits < top_logits[:, -1][:, None] # Get the indices to remove
            final_logits = final_logits.masked_fill(indices_to_remove, -float('inf')) # Mask
        elif top_p is not None:
            assert 1.0 >= top_p > 0.0, "top_p must be in [0.0, 1.0)" # Assert that top_p is in [0.0, 1.0)
            sorted_logits, sorted_indices = torch.sort(final_logits, descending=True) # Sort the logits and indices
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # Get the cumulative probabilities
            sorted_indices_to_remove = cumulative_probs > top_p # Get the sorted indices to remove. We round up since we want prob >= top_p not < top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # Shift the indices to the right
            sorted_indices_to_remove[..., 0] = 0  # Ensure at least one token is kept
            indices_to_remove = sorted_indices_to_remove.scatter( # Scatter the sorted indices to remove
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            final_logits = final_logits.masked_fill(indices_to_remove, -float('inf')) # Mask
        return torch.distributions.categorical.Categorical(logits=final_logits).sample() # Sample from the distribution
    

# Type alias
SliceInput = Optional[Union[int, Tuple[int, int], Tuple[int, int, int], List[int], torch.Tensor]]
class Slice:
    '''
    Use a customer slice syntax since Pthon/Torch's do not let us reduce the number of dimensions.
    Note that slicing with input_slice=None means do nothing, NOt add an extra dimension (use unsqueeze for that),
    
    There are several modes:
    int - just index with that integer (decreases number of dimensions)
    slice - Input is a tuple converted to a slce ((k,) means :k, (k,m) means m:k, (k,m,n) means m:k:n)
    array - Input is a list or tensor or numpy array, converted to a Numpy array, and we take the stack of values at those indices
    identity - Input is None, do nothing.

    Examples for dim=0:
    if input_slice=0, tensor -> tensor[0]
    elif input_slice = (1, 5), tensor -> tensor[1:5]
    elif input_slice = (1, 5, 2), tensor -> tensor[1:5:2] (ie indexing with [1, 3])
    elif input_slice = [1, 4, 5], tensor -> tensor[[1, 4, 5]] (ie changing the first axis to have length 3, and taking the indices 1, 4, 5 out).
    elif input_slice is a Tensor, same as list - Tensor is assumed to be a 1D list of indices.

    '''
    def __init__(
            self, 
            input_slice: SliceInput = None,
    ):
        if type(input_slice)==tuple:
            input_slice = slice(*input_slice) # Convert tuple to slice
            self.slice = input_slice
            self.mode='slice'
        elif type(input_slice)==int:
            self.slice = input_slice
            self.mode='int'
        elif type(input_slice)==slice:
            self.slice = input_slice
            self.mode='slice'
        elif type(input_slice)==list or type(input_slice)==torch.Tensor or type(input_slice)==np.ndarray:
            self.slice = np.array(input_slice) # Convert to numpy array
            self.mode='array'
        elif input_slice is None:
            self.slice = None
            self.mode='identity'
        else:
            raise ValueError(f"Invalid input_slice type: {type(input_slice)}")
    
    def apply(self, tensor: Tensor,dim=0) -> Tensor:
        '''
        Applies the slice to the tensor.
        Args:
            tensor (Tensor): The tensor to slice.
        Returns:
            Tensor: The sliced tensor.
        '''
        ndim = tensor.ndim # Get the number of dimensions
        slices = [slice(None)] * ndim # Create a list of slices
        slices[dim] = self.slice # Set the slice for the specified dimension
        return tensor[tuple(slices)] # Return the sliced tensor
    
    def indices(self, max_ctx=None):
        '''
        Returns the indices of the slice.
        Args:
            max_ctx (int, optional): The maximum context length. Required for slice mode. Defaults to None.
        Returns:
            np.ndarray: The indices of the slice.
        '''
        if self.mode=='int':
            return np.array([self.slice])
        else:
            return np.arange(max_ctx)[self.slice]
    
    def __repr__(self):
        return f"Slice(mode={self.mode}, slice={self.slice})"
    
def act_name(
        name:str,
        layer: Optional[int]=None,
        layer_type: Optional[str]=None,
    ) -> str:
    '''
    Helper function to convert shorthand to an activation name. 
    Args:
        name (str): The name of the activation.
        layer (Optional[int], optional): The layer number. Defaults to None.
        layer_type (Optional[str], optional): The layer type. Defaults to None.
    Returns:
        str: The full activation name.
    Examples:
        act_name('k', 6, 'a')=='blocks.6.attn.hook_k'
        act_name('pre', 2)=='blocks.2.mlp.hook_pre'
        act_name('embed')=='hook_embed'
        act_name('normalized', 27, 'ln2')=='blocks.27.ln2.hook_normalized'
        act_name('k6')=='blocks.6.attn.hook_k'
        act_name('scale4ln1')=='blocks.4.ln1.hook_scale'
        act_name('pre5')=='blocks.5.mlp.hook_pre'
    '''
    match = re.match(r"[a-z]+)(\d+)([a-z]?.*)", name)
    if match is not None:
        name, layer, layer_type = match.groups(0)
    layer_type_dict = {'a':'attn', 'm':'mlp', 'b':'','block':'','blocks':'','attention':'attn'}
    act_name = ""
    if layer is not None:
        act_name += f"blocks.{layer}."
    if name in ['k','q','v','result','attn','attn_scores']:
        layer_type = 'attn'
    elif name in ['pre','post','mid']:
        layer_type = 'mlp'
    if layer_type in layer_type_dict:
        layer_type = layer_type_dict[layer_type]
    
    if layer_type:
        act_name += f"{layer_type}."
    act_name += f"hook_{name}"
    return act_name

def transpose(tensor):
    '''
    Transposes the last two dimensions of a tensor.
    Args:
        tensor (Tensor): The tensor to transpose.
    Returns:
        Tensor: The transposed tensor.

    '''
    return tensor.transpose(-1, -2) # Transpose the last two dimensions


def composition_scores(
        left: FactoredMatrix,
        right: FactoredMatrix,
        broadcast_dims: bool = True,
):
    '''
    Obtains the composition scores between two factored matrices.
    Args:
        left (factored_matrix.FactoredMatrix): The left factored matrix.
        right (factored_matrix.FactoredMatrix): The right factored matrix.
        broadcast_dims (bool, optional): Whether to broadcast the dimensions. Defaults to True.
    Returns:
        Tensor: The composition scores.
    '''
    if broadcast_dims:
        r_leading = right.ndim-2 # Get the number of leading dimensions for the right factored matrix
        l_leading = left.ndim-2 # Get the number of leading dimensions for the left factored matrix
        for i in range(l_leading):
            right = right.unsqueeze(i) # Unsqueeze the right factored matrix in the ith dimension
        for i in range(r_leading):
            left = left.unsqueeze(i+l_leading) # Unsqueeze the left factored matrix in the (i+l_leading)th dimension
    assert left.rdim==right.ldim, f"Composition scores require left.rdim==right.ldim, got {left.rdim} and {right.ldim}" # Assert that the left factored matrix's rdim is equal to the right factored matrix's ldim
    right = right.collapse_r() # Collapse the right factored matrix
    left = left.collapse_l() # Collapse the left factored matrix
    r_norms = right.norm(dim=[-2,-1]) # Get the norms of the right factored matrix
    l_norms = left.norm(dim=[-2,-1]) # Get the norms of the left factored matrix
    comp_norms = (left @ right).norm(dim=[-2,-1]) # Get the norms of the composition
    return comp_norms/r_norms/l_norms # Return the composition scores


def verify_activations(model: Any,
                       text:str) -> bool:
    '''
    Verifies activations in the attention matrix.
    Args:
        model (Any): a model.
        text (str): A string of text.
    Returns:
        A boolean of whether the layer 0 patter from Q and K is close to the layer 0 pattern from cache or not. 
    '''
    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    layer_0_pattern_from_cache = model["pattern", 0]
    Q = cache["q",0] # Obtain the Q (query) vectors from cache 
    K = cache["k",0] # Obtain the K (key) vectors from cache 
    seq, n_head, d_K = K.shape
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    Q_K_attention = einops.einsum(Q, K, "... n h, ... n h -> n ...")
    mask = torch.triu(torch.ones((seq, seq),dtype=torch.bool), diagonal=1).to(device) # Create a mask by creating the seq x seq matrix first 
    
    layer0_masked_attn = Q_K_attention.masked_fill_(mask, -1e9) # Fill the attention patterns from Q and K with mask and near-zero matrix
    softmaxed = (layer0_masked_attn / d_K** 0.5).softmax(-1) # Scale the masked attention by the square root of the number of heads d_K and apply the softmax function to the masked attention  
    return torch.testing.assert_close(layer_0_pattern_from_cache,softmaxed) # Test to see if the attention patterns from layer 0 from cache and the softmaxed attention patterns from Q and K are numerically close
    

            