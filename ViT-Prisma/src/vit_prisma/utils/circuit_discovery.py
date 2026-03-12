from copy import deepcopy
from functools import partial
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import torch
from vit_prisma.utils.experiments import get_act_hook
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from collections import OrderedDict
from vit_prisma.utils.ioi_utils import show_pp
import graphviz 

#def get_hook_tuple(layer: int,
#                   head_idx: int,
#                   comp: bool=None, 
#                   input: bool=False,):
    
