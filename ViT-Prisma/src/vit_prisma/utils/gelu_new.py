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
