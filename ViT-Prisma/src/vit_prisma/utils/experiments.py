# %%
from typing import Callable, Union, List, Tuple, Any
import torch
import warnings
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
import random
import time
from pathlib import Path
import pickle
import os
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from functools import *
import pandas as pd
import gc 
import collections
import copy
import itertools
from transformers import AutoModelForObjectDetection, AutoConfig, AutoTokenizer
from vit_prisma.prisma_tools.hook_point import HookPoint

# %%
def get_act_hook(
        fn,
        alt_act: Any=None,
        idx: Any = None,
        dim: Any = None,
        name: Any = None,
        message: Any = None,
        metadata: Any = None, 

):
    '''
    Returns a hook that modifies the activation on the fly. 
    Args:
        fn: Function.
        alt_act: Alternative activations (tensor of the same shape of z).
        idx: Index of the head.
        dim: Dimension of the tensor.
        name: Name of the activation.
        message: Message.
        metadata: metadata of the activation. 
    Returns: 
        Tensor
    '''
    if alt_act is not None:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name
            hook.ctx["metadata"] = metadata

            if message is not None:
                print(message)
            
            if (
                dim is None
            ): # Mean and z have the same shape, the mean is constant along the batch dimension
                return fn(z,alt_act,hook)
            if dim == 0:
                z[idx] = fn(z[idx], alt_act[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], alt_act[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], alt_act[:, :, idx], hook)
            return z
        
    else:

        def custom_hook(z, hook):
            '''
            Create custom hooks. 
            '''
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name
            hook.ctx["metadata"] = metadata
        
            if message is not None:
                print(message)
            
            if dim is None:
                return fn(z, hook)
            if dim == 0:
                z[idx] = fn(z[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], hook)
            return z
        
    return custom_hook

