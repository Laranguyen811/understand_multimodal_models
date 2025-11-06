from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.sae.evals.evals import (get_substitution_loss, get_text_embeddings_openclip, 
                                        get_feature_probability)
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.dataloaders.imagenet_index import imagenet_index
from vit_prisma.transforms.model_transforms import get_model_transforms
import matplotlib.pyplot as plt

import open_clip

import numpy as np
from statistics import mean
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm

import einops

import numpy as np
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

from vit_prisma.sae.evals.evals import zero_ablate_hook

from typing import Any, List, Tuple, Dict
from functools import partial

import logging

from torch.utils.data import Subset

from torch.cuda.amp import autocast

logging.basicConfig(
    level=logging.DEBUG,  # or INFO for less verbosity
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

class LinearClassifier(nn.Module):
    """A simple linear layer on top of frozen features."""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    if url is not None:
        print("Loading pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )["state_dict"]
        new_state_dict = {
            'linear.weight': state_dict['module.linear.weight'],
            'linear.bias': state_dict['module.linear.bias']
        }
        linear_classifier.load_state_dict(new_state_dict, strict=True)
    else:
        print("Using random linear weights.")



class SparsecoderEval():

    def __init__(self, sc: SparseAutoencoder, model: HookedViT, dataset: DatasetFolder, 
                 batch_size=4, max_images=None):
        
        self.sc = sc
        self.model = model
        self.is_transcoder = self.sc.cfg.is_transcoder

        self.dtype = None
        self.max_images = max_images

        self.hook_point_filters = [sc.cfg.hook_point]
        if self.is_transcoder:
            self.hook_point_filters.append(sc.cfg.out_hook_point)

        data_transforms = get_model_transforms(self.model.cfg.model_name)

        self.batch_size = batch_size

        if max_images is not None:
            indices = list(range(min(max_images, len(dataset))))
            self.validation_dataset = Subset(dataset, indices)
        else:
            self.validation_dataset = dataset

        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        if 'dino' in self.model.cfg.model_name:
            self.is_dino = True
            self.classifier_head = LinearClassifier(1536, num_labels=1000).cuda()
            load_pretrained_linear_weights(self.classifier_head, 'vit', 16)
        else:
            self.is_dino = False


    def run_eval(self, is_clip=False):

        all_l0 = []
        all_l0_cls = []

        # image level l0
        all_l0_image = []

        total_loss = 0
        total_score = 0
        total_reconstruction_loss = 0
        total_zero_abl_loss = 0
        total_samples = 0
        all_cosine_similarity = []
        all_recons_cosine_similarity = []

        self.model.eval()
        self.sc.eval()

        if is_clip:
            num_imagenet_classes = 1000
            batch_label_names = [imagenet_index[str(int(label))][1] for label in range(num_imagenet_classes)]

            model_name = self.model.cfg.model_name
            repo_path = model_name.split(':', 1)[1]  # Get everything after the first ':'
            new_model_id = f'hf-hub:{repo_path}'

            og_model, _, preproc = open_clip.create_model_and_transforms(new_model_id)
            tokenizer = open_clip.get_tokenizer(new_model_id)

            # download huggingface

            text_embeddings = get_text_embeddings_openclip(og_model, preproc, tokenizer, batch_label_names)
            del og_model, tokenizer

        total_acts = None
        total_tokens = 0
        total_images = 0

        total_batches = 0

        logger.info("Starting SparseCoder evaluation...")

        with torch.no_grad():
            pbar = tqdm(self.validation_dataloader, desc="Evaluating", dynamic_ncols=True)

            num_features = self.sc.W_enc.shape[1]

            total_patches = 0
            feature_activation_counts = torch.zeros(num_features, device='cuda')
            feature_activation_sums = torch.zeros(num_features, device='cuda')

            with autocast():
                for _, batch in enumerate(pbar):
                    batch_tokens, gt_labels = batch
                    batch_tokens = batch_tokens.to(self.sc.cfg.device)

                    if self.dtype == torch.float16:
                        batch_tokens = batch_tokens.half()

                    batch_size = batch_tokens.shape[0]
                    # batch shape
                    total_samples += batch_size
                    output, cache = self.model.run_with_cache(batch_tokens, names_filter=self.hook_point_filters)

                    if getattr(self.sc.cfg, "hook_point_filters", None) == "output":
                        hook_point_activation = output.unsqueeze(1) # Unsqueeze to give output "patch"" dimension
                    else:
                        hook_point_activation = cache[self.hook_point_filters[0]].to(self.sc.cfg.device)

                    if self.sc.cfg.use_patches_only:
                        hook_point_activation = hook_point_activation[:,1:,:]
                    elif self.sc.cfg.cls_token_only:
                        hook_point_activation = hook_point_activation[:,0:1,:]
                    
                    args = [hook_point_activation]
                    if self.is_transcoder:
                        out_hook_point_activation = cache[self.hook_point_filters[1]].to(self.sc.cfg.device)
                        args.append(out_hook_point_activation)

                    sae_out, feature_acts, loss, mse_loss, l1_loss, _, aux_loss = self.sc(*args)


                    batch_size, seq_len, *rest = feature_acts.shape
                    total_patches += batch_size * seq_len

                    feature_acts_flat = feature_acts.reshape(-1, num_features)
                    feature_active = (feature_acts_flat > 0).float()
                    feature_activation_counts += feature_active.sum(dim=0)
                    feature_activation_sums += feature_acts_flat.sum(dim=0)

                    sae_activations = get_feature_probability(feature_acts)
                    if total_acts is None:
                        total_acts = sae_activations.sum(0)
                    else:
                        total_acts += sae_activations.sum(0)
                    
                    total_tokens += sae_activations.shape[0]
                    total_images += batch_size

                    l0 = (feature_acts > 0).float().sum(-1).detach()

                    all_l0.extend(l0.mean(dim=1).cpu().numpy())
                    l0_cls = (feature_acts[:, 0, :] > 0).float().sum(-1).detach()
                    all_l0_cls.extend(l0_cls.flatten().cpu().numpy())

                    l0 = (feature_acts > 0).float().sum(-1).detach()
                    
                    l0_per_sample = [
                        feature_acts[i].nonzero(as_tuple=True)[1].unique().numel()
                        for i in range(feature_acts.shape[0])
                    ]
                    all_l0_image.extend(l0_per_sample)

                    cos_sim = torch.cosine_similarity(einops.rearrange(hook_point_activation, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                                    einops.rearrange(sae_out, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                                        dim=0).mean(-1).tolist()
                    all_cosine_similarity.append(cos_sim)

                    if is_clip:
                        score, loss, recons_loss, zero_abl_loss = get_substitution_loss(self.sc, self.model, batch_tokens, gt_labels, 
                                                                                text_embeddings)
                    elif self.is_dino:

                        class_logits = self.classifier_head(model(batch_tokens))

                        loss = F.cross_entropy(class_logits.cuda(), gt_labels.cuda())

                        head_index = self.sc.cfg.hook_point_head_index
                        hook_point = self.sc.cfg.out_hook_point if self.is_transcoder else self.sc.cfg.hook_point
                        
                        def standard_replacement_hook(activations: torch.Tensor, hook: Any):
                            activations = self.sc.forward(activations)[0].to(activations.dtype)
                            return activations

                        def head_replacement_hook(activations: torch.Tensor, hook: Any):
                            new_activations = self.sc.forward(activations[:, :, head_index])[0].to(activations.dtype)
                            activations[:, :, head_index] = new_activations
                            return activations

                        replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook
            
                        if getattr(self.sc.cfg, "hook_point_filters", None) == "output":
                            output = self.model(batch_tokens)
                            recons_image_embeddings = self.sc(output)
                        else:
                            activations = self.model.run_with_hooks(
                                batch_tokens,
                                fwd_hooks=[(hook_point, partial(replacement_hook))],
                            )
                            recons_image_embeddings = self.classifier_head(activations)

                        recons_loss = F.cross_entropy(recons_image_embeddings.cuda(), gt_labels.cuda())

                        zero_abl_image_embeddings = self.classifier_head(model.run_with_hooks(
                            batch_tokens, fwd_hooks=[(hook_point, zero_ablate_hook)]
                        ))

                        zero_abl_loss = F.cross_entropy(zero_abl_image_embeddings.cuda(), gt_labels.cuda())

                        score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)


                    total_loss += loss.item()
                    total_score += score.item()
                    total_reconstruction_loss += recons_loss.item()
                    total_zero_abl_loss += zero_abl_loss.item()
                    # all_recons_cosine_similarity.extend(recons_cosine_sim)

                    pbar.set_postfix({
                        'L0': f"{mean(l0_per_sample)}",
                        'Cosine Sim': f"{cos_sim:.6f}"
                    })

                    total_batches += 1

                
                    if self.max_images is not None and total_images > self.max_images:
                        break

                    
        
        logger.info("Finished running through validation dataset...")
        logger.info("Computing metrics...")
                
        feature_activation_counts.to('cpu')
        feature_activation_sums.to('cpu')

        denom = total_batches

        avg_loss = total_loss / denom
        avg_reconstruction_loss = total_reconstruction_loss / denom
        avg_zero_abl_loss = total_zero_abl_loss / denom
        avg_score = total_score / denom


        ce_recovered =  ((avg_zero_abl_loss - avg_reconstruction_loss) / ((avg_zero_abl_loss - avg_loss) + 1e-6)) * 100
        
        avg_l0 = np.mean(all_l0)
        avg_l0_cls = np.mean(all_l0_cls)
        avg_l0_image = np.mean(all_l0_image)

        avg_cos_sim = np.mean(all_cosine_similarity)
        # total_recons_cosine_similarity = np.mean(all_recons_cosine_similarity)

        metrics = {}
        metrics['avg_l0'] = avg_l0.astype(float)
        metrics['avg_cls_l0'] = avg_l0_cls.astype(float)
        metrics['avg_image_l0'] = avg_l0_image.astype(float)
        metrics['avg_cosine_similarity'] = avg_cos_sim.astype(float)
        # metrics['avg_recons_cosine_similarity'] = total_recons_cosine_similarity.astype(float)
        metrics['avg_CE'] = avg_loss
        metrics['avg_recons_CE'] = avg_reconstruction_loss
        metrics['avg_zero_abl_CE'] = avg_zero_abl_loss
        metrics['CE_recovered'] = ce_recovered

        # print out everything above
        logger.info(f"Average L0 (features activated): {avg_l0:.6f}")
        logger.info(f"Average L0 (features activated) per CLS token: {avg_l0_cls:.6f}")
        logger.info(f"Average L0 (features activated) per image: {avg_l0_image:.6f}")
        logger.info(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
        logger.info(f"Average Loss: {avg_loss:.6f}")
        logger.info(f"Average Reconstruction Loss: {avg_reconstruction_loss:.6f}")
        logger.info(f"Average Zero Ablation Loss: {avg_zero_abl_loss:.6f}")
        logger.info(f"Average CE Score: {avg_score:.6f}")
        logger.info(f"% CE recovered: {ce_recovered:.6f}")
        # print(f"Average Reconstruction Cosine Similarity: {total_recons_cosine_similarity:.6f}")
        

        return metrics
