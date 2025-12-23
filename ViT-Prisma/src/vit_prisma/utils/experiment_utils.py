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
from typing import Any

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

def calculate_gelu_fast(input):
    '''
    Implementation of the gelu activation function (fast version).
    '''
    return 0.5 * input * (1 + torch.tanh(0.7978845608 * (1 + 0.044715 * input ** 2)))

def calculate_solu(input):
    '''
    Implementation of the solu (Softmax Linear Unit) activation function.
    LayerNorm implemented by the MLP class. 
    '''
    return input * F.softmax(input, dim=-1) 

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
        tokeniser.encode_plus({'pad_token': '[PAD]'}) # Add a pad token
