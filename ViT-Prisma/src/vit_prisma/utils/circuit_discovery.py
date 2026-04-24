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
from vit_prisma.utils import detect_architectures
from torch import Tensor
from vit_prisma.utils.patching_utils import direct_path_patching, direct_path_patching_up_to

def get_hook_tuple(model:Any,
                   layer: int,
                   head_idx: int,
                   comp: str=None, 
                   input: bool=False,
                   n_layers: int=12) -> Tuple[str, Callable]:
    '''
    Helper function to create a hook tuple for a given layer and head index.
    Args:
        model (Any): The model for which to create the hook tuple.
        layer (int): The layer number for which to create the hook tuple.
        head_idx (int): The head index for which to create the hook tuple.
        comp (str, optional): The computation for which to create the hook tuple. Defaults to None.
        input (bool, optional): Whether to include the input in the hook tuple. Defaults to False.
        n_layers (int, optional): The total number of layers in the model. Defaults to 12.
    Returns:
        Tuple[str, Callable]: A tuple containing the hook name and the hook function.
    '''
    arch = detect_architectures(model)
    HOOKS = {}

    if comp in ["q", "k", "v"]:
        hook_pattern = HOOKS[arch][comp]
        assert comp in ["q", "k", "v"], f"Invalid comp: {comp}. Must be one of 'q', 'k', or 'v'."
        assert head_idx is not None, f"head_idx must be provided when comp is one of 'q', 'k', or 'v'."
        return hook_pattern.format(layer=layer, head_idx=head_idx)
    elif head_idx is None:
        if layer < n_layers:    
            hook_pattern = HOOKS[arch]["mlp_out" if not input else "resid_mid"] 
        else:
            hook_pattern = HOOKS[arch]["resid_post"]
        return hook_pattern.format(layer=layer, head_idx=head_idx)

def patch_all(z: Tensor,
              source_act: Tensor,
               hook: Tensor,) -> Tensor:
    '''
    Patch the activations of a model with the activations from a source model.
    Args:
        z (Tensor): The activations of the model to be patched.
        source_act (Tensor): The activations from the source model to patch with.
        hook (Tensor): The hook tensor indicating where to patch.
        Returns:
        Tensor: The patched activations.
    '''
    z[:] = source_act[hook]
    return z

def patch_positions(
        z: Tensor,
        source_act: Tensor,
        hook: Tensor,
        positions: List[int],
) -> Tensor:
    '''
    Patch the activations of a model at specific positions with the activations from a source model.
    Args:
        z (Tensor): The activations of the model to be patched.
        source_act (Tensor): The activations from the source model to patch with.
        hook (Tensor): The hook tensor indicating where to patch.
        positions (List[int]): The list of positions to patch.
    Returns:
        Tensor: The patched activations.
    '''
    if positions is None: # Same as patch all
        raise NotImplementedError(
            "Patching all positions is not implemented in patch_positions. Use patch_all instead."
        )
    else:
        batch = z.shape[0]
        cur_positions = Tensor(positions)
        if len(cur_positions.shape) == 0:
            cur_positions = cur_positions.unsqueeze(0)
        for pos in cur_positions: 
            z[torch.arange(batch), pos] = source_act[torch.arange(batch), pos]
        return z

def get_datasets(text: str,
                 dataset: Any) -> Tuple[Tensor, Tensor]:
    '''
    Get the datasets for the circuit discovery experiments.
    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the input and output datasets.
    '''
    batch_size = 1
    orig = text
    output = dataset(orig, return_tensors="pt", padding=True)
    obj = [obj for obj in dataset if obj != output]

    return orig, output,obj

def old_path_patching(
        model: Any,
        orig_data: Tensor,
        new_data: Tensor,
        senders: List[Tuple],
        receiver_hooks: List[Tuple],
        max_layer: Union[int, None] = None,
        position: int = 0,
        return_hooks: bool = False,
        freeze_mlps: bool = True,
        orgin_cache: Optional[Dict] = None,
        new_cache: Optional[Dict] = None,
        prepend_bos: bool = False,

) -> Union[Tensor, Tuple[Tensor, Dict, Dict]]:
    '''
    Perform path patching on a model by patching the activations from a source model to a target model in the old way.
    Args:
        model (Any): The model to perform path patching on.
        orig_data (Tensor): The original input data for the model.
        new_data (Tensor): The new input data for the model.
        senders (List[Tuple]): A list of tuples containing the sender layer and head index for patching.
        receiver_hooks (List[Tuple]): A list of tuples containing the receiver layer and head index for patching.
        max_layer (Union[int, None], optional): The maximum layer to consider for patching.
        position (int, optional): The position to patch. Defaults to 0.
        return_hooks (bool, optional): Whether to return the hooks used for patching. Defaults to False.
        freeze_mlps (bool, optional): Whether to freeze the MLPs during patching. Defaults to True.
        orgin_cache (Optional[Dict], optional): The cache for the original model. Defaults to None.
        new_cache (Optional[Dict], optional): The cache for the new model. Defaults to None.
        prepend_bos (bool, optional): Whether to prepend a beginning of sequence token to the input data. Defaults to False.
    '''
    if max_layer is None:
        max_layer = model.cfg.n_layers
    assert max_layer <= model.cfg.n_layers, f"max_layer must be less than or equal to the number of layers in the model ({model.cfg.n_layers})."

    if orig_cache is None:

        # Save activations from original data
        orig_cache = {}
        model.reset_hooks()
        model.cache_all (orig_cache)
        _ = model(orig_data, prepend_bos=False)

    # Process senders
    senders_hooks = []
    for layer, head in senders:
        if head is not None: #(layer, head) for attention heads
            senders_hooks.append((f"blocks.{layer}.attn.hook_result", head))
        else: # (layer, None) for MLPs
            senders_hooks.append((f"blocks.{layer}.hook_mlp_out", None))
    sender_hook_names = [x[0] for x in senders_hooks]

    if new_cache is None:
        # Save activations from new for senders
        model.reset_hooks()
        new_cache = {}
        model.cache_some(new_cache, lambda x: x in sender_hook_names)
        _ = model(new_data, prepend_bos=False)

    else:
        assert all(
            [x in new_cache for x in sender_hook_names]
        ), f"Difference between new_cahce and senders: {set(sender_hook_names) - set(new_cache.keys())}"

    # Configure hooks for freezing activations
    for layer in range(max_layer):
        # heads
        for head in range(model.cfg.n_heads):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)
                hook = get_act_hook(
                    patch_all,
                    alt_act=orig_cache[hook_name],
                    idx=head,
                    dim=2,
                )
                model.add_hook(hook_name, hook)
    
    # MLPs
    if freeze_mlps:
        hook_name = f"blocks.{layer}.hook_mlp_out"
        hook = get_act_hook(
            patch_all,
            alt_act=orig_cache[hook_name],
            idx=None,
            dim=None,
        )
        model.add_hook(hook_name, hook)
    
    # For senders, add new hook to patching new activations
    for hook_name, head in senders_hooks:
        hook = get_act_hook(
            partial(patch_positions, positions=[position]),
            alt_act=new_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        model.add_hook(hook_name, hook)

    # Forward pass on origin, where patch in new activations for senders and original activations for all other hooks (including receiver_hooks). 
    # Save activations on receivers

    _ = model(orig_data, prepend_bos=False)
    model.reset_hooks()

    # Add hooks for final forward pass on origin data, where we patch in hybrid activations for receivers
    hooks = []
    for hook_name, head in receiver_hooks:
        hook = get_act_hook(
            partial(patch_positions, positions=[position]),
            alt_act=orig_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        hooks.append((hook_name, hook))

    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model
    
class Node:
    def __init__(self,
                layer:int,
                head: int,
                position: str):
        self.layer = layer
        self.head = head
        assert isinstance(position, str), f"Position must be a string, not{type(position)}."
        self.position = position
        self.children = []
        self.parents = []

    def __repr__(self):
        '''
        String representation of the Node object.
        '''
        return f"Node(layer={self.layer}, head={self.head}, position={self.position})"

    def repr_long(self):
        '''
        Long string representation of the Node object, including its children and parents.
        '''
        return f"Node(layer={self.layer}, head={self.head}, position={self.position} with children={[child.__repr__() for child in self.children]}, parents={[parent.__repr__() for parent in self.parents]})"

    def display(self):
        '''
        Display the node and its children in a readable format.
        '''
        if self.layer == 12:
            return "resid out"

        elif self.head is not None:
            return f"mlp{self.layer}\n{self.position}"
        
        else:
            return f"{self.layer}.{self.head}\n{self.position}"

class HypothesisTree:
    def __init__(
            self,
            model: Any,
            metric: Callable,
            orig_data: Tensor,
            new_data: Tensor,
            threshold: float,
            orig_positions: OrderedDict,
            new_positions: OrderedDict,
            dataset: Any=None,
            use_caching: bool=False,
            direct_paths_only: bool=False,
    ):
        model.reset_hooks()
        self.model = model
        self.orig_positions = orig_positions
        self.new_positions = new_positions
        assert list(orig_positions.keys()) == list(new_positions.keys()), "Number and order of keys should be the same for now."
        self.node_stack = OrderedDict()
        self.populate_node_stack()
        self.current_node = self.node_stack[
            next(reversed(self.node_stack))
        ] # last element
        self.root_node = self.current_node
        self.metric = metric
        self.dataset = dataset
        self.orig_data = orig_data
        self.new_data = new_data
        self.threshold = threshold
        self.default_metric = self.metric(model,dataset)
        self.orig_cache = None
        self.new_cache = None
        self.use_caching = True
        if use_caching:
            self.get_caches()
        self.important_nodes = []
        self.direct_paths_only = direct_paths_only

    def populate_node_stack(self):
        '''
        Populate the node stack with all possible nodes based on the original and new positions.
        '''
        for layer in range(self.model.cfg.n_layers):
            for head in list(range(self.model.cfg.n_heads)) + [None]:
                # Includes None for MLP
                for pos in self.orig_positions:
                    node = Node(layer, head, pos)
                    self.node_stack[(layer, head, pos)] = node
            layer = self.model.cfg.n_layers
            pos = next(reversed(self.orig_positions)) # Assume the last position specified is the one that we care about in the residual stream
            resid_post = Node(layer, None, pos)

    def get_caches(self):
        '''
        Get the caches for the original and new data.
        '''
        if "orgin_cache" in self.__dict__.keys():
            warnings.warn("Overwriting existing origin cache.")
        
        # Save activations from original data
        self.orig_cache = {}
        self.model.reset_hooks()
        self.model.cache_all (self.orig_cache)
        _ = self.model(self.orig_data, prepend_bos=False)
    
    def eval(
            self,
            threshold: Union[float, None] = None,
            verbose: bool = False,
            show_graphics: bool = False,
            auto_threshold: float = 0.0,

    )-> None:
        '''
        Evaluate the hypothesis tree by performing path patching and checking the metric against the threshold.
        Args:
            threshold (Union[float, None], optional): The threshold for determining whether a node is important. If None, uses the default threshold. Defaults to None.
            verbose (bool, optional): Whether to print verbose output during evaluation. Defaults to False.
            show_graphics (bool, optional): Whether to show graphics of the hypothesis tree during evaluation. Defaults to False.
            auto_threshold (float, optional): The automatic threshold for determining whether a node is important based on the metric change. Defaults to 0.0.
        Returns:
            None
        '''
        if self.direct_paths_only:
            print("1")
            self.eval_new(
                threshold=threshold,
                verbose=verbose,
                show_graphics=show_graphics,
                auto_threshold=auto_threshold,
            )
        
        else:
            print("2")
            self.eval_old(
                threshold=threshold,
                verbose=verbose,
                show_graphics=show_graphics,
                auto_threshold=auto_threshold,
            )
        
    def eval_new(
                self,
                threshold: Union[float, None] = None,
                verbose: bool = False,
                show_graphics: bool = False,
                auto_threshold: float = 0.0,
        )-> None:
        '''
        Evaluate the hypothesis tree using the direct path patching method.
        '''
        if threshold is None:
            threshold = self.threshold
        _, node = self.node_stack.popitem() # Remove the item last inserted into the node stack
            
        self.important_nodes.append(node)
        print(f"Currently evaluating node {node}")

        current_node_position = node.position
        for pos in self.orig_positions:
            if (
                current_node_position != pos and node.head is None
            ):
                # MLPs and the end state of the residual stream only care about the last position
                continue
        
        receiver_hooks = []
        if node.layer == self.model.cfg.n_layers:
            receiver_hooks.append((f"blocks.{node.layer-1}.hook_resid_post", None))
        elif node.head is None:
            receiver_hooks.append((f"blocks.{node.layer}.hook_resid_mid", None))
        else:
            receiver_hooks.append((f"blocks.{node.layer}.attn.hook_v_output", node.head))
            receiver_hooks.append((f"blocks.{node.layer}.attn.hook_k_input",node.head))
            if pos == current_node_position:
                receiver_hooks.append((f"blocks.{node.layer}.attn.hook_q_input", node.head)) # Only the last position that we care about, similarly

        for receiver_hook in receiver_hooks:
            print(f"Working on pos {pos}, receiver hook {receiver_hook}")

            attn_results, mlp_results = direct_path_patching_up_to(
                model=self.model,
                receiver_hook=receiver_hook,
                important_nodes=self.important_nodes,
                metric=self.metric,
                dataset=self.dataset,
                orig_data=self.orig_data,
                new_data=self.new_data,
                orig_positions=self.orig_positions,
                new_positions=self.new_positions,
                cur_position=pos,
                orig_cache=self.orig_cache,
                new_cache=self.new_cache,
            )

            self.attn_results = attn_results.clone() # Clone attn_results to avoid modifying actual attn_results 
            self.mlp_results = mlp_results.clone() # Clone attn_results to avoid modifying actual mlp_results

            # Convert to percentage
            attn_results -= self.default_metric # Subtract default_metric from attn_results
            self.mlp_results /= self.default_metric # Divide mlp_results by default_metric
            mlp_results -= self.default_metric # Subtract default_metric from mlp_results
            mlp_results /= self.default_metric # Divide mlp_results by default_metric

            if show_graphics:
                show_pp(
                    attn_results,
                    title=f"{node} {receiver_hook} {pos}",
                    xlabel="Head",
                    ylabel="Layer",
                )
                show_pp(
                    mlp_results,
                    title=f"MLP results for {node} with receiver hook {receiver_hook} position {pos}",
                    xlabel="Layer",
                    ylabel="",
                )

                if auto_threshold:
                    threshold = max(
                        auto_threshold * attn_results.std(),
                        auto_threshold * mlp_results.std(),
                        0.01,
                    )

                if verbose:
                    print(f"threshold: {threshold:.3f}")
                # Process results and mark nodes above threshold as important
                for layer in range(
                    attn_results.shape[0]
                ):
                    for head in range(attn_results.shape[1]):
                        if abs(attn_results[layer,head]) > threshold:
                            print(
                                f"Found important head {(head, layer)} at position {pos} with change {attn_results[layer, head]:.3f} in metric."
                            )
                            score = attn_results[layer,head],
                            comp_type = receiver_hook[0].split("_")[-1] # "q_input", "k_input", "v_output", "resid_mid", or "resid_post"
                            if (
                                comp_type == "input"
                            ):
                                for letter in "qkv":
                                    if f"_{letter}_" in receiver_hook[0]:
                                        comp_type = letter
                                assert comp_type in ["q", "k", "v"], f"Invalid comp_type: {comp_type}, received hook: {receiver_hook[0]}"

                            self.node_stack[(layer,head,pos)].parents.append((node, score, comp_type))
                            node.children.append((self.node_stack[(layer,head,pos)], score, comp_type))
                        if(
                            layer < mlp_results.shape[0] and abs(mlp_results[layer]) > threshold
                        ):
                            print(
                                f"Found important MLP {layer} at position {pos} with change {mlp_results[layer]:.3f} in metric."
                            )
                            score = mlp_results[layer]
                            comp_type = receiver_hook[0].split("_")[-1] # "q_input", "k_input", "v_output", "resid_mid", or "resid_post"

                    


        
    
