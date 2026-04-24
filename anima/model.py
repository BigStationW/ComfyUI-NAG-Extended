from functools import partial
from types import MethodType

import torch

from .layers import nag_self_attn_forward
from ..utils import cat_context, check_nag_activation, NAGSwitch


def forward_nag_anima(
        self,
        x,
        timestep,
        context,
        y=None,
        **kwargs,
):
    """
    NAG wrapper for Anima / Cosmos forward.
    Dynamically expands the batch to process positive and negative paths together.
    """
    transformer_options = kwargs.get("transformer_options", {})
    nag_sigma_start = kwargs.get("nag_sigma_start", 14.7)
    nag_sigma_end = kwargs.get("nag_sigma_end", 0.0)
    nag_negative_context = kwargs.get("nag_negative_context", None)
    nag_negative_y = kwargs.get("nag_negative_y", None)
    
    apply_nag = check_nag_activation(transformer_options, nag_sigma_start, nag_sigma_end)
    
    if apply_nag and nag_negative_context is not None:
        pos_bsz = x.shape[0]
        nag_bsz = nag_negative_context.shape[0]
        
        # 1. Concatenate text contexts
        context_extended = cat_context(context, nag_negative_context, trim_context=True)
        
        # 2. Expand image latents (x) and timesteps to match the new batch size
        if nag_bsz > pos_bsz:
            repeat_times = (nag_bsz + pos_bsz - 1) // pos_bsz
            x_neg = x.repeat(repeat_times, 1, 1, 1)[:nag_bsz]
            t_neg = timestep.repeat(repeat_times)[:nag_bsz]
        else:
            x_neg = x[:nag_bsz]
            t_neg = timestep[:nag_bsz]
            
        x_extended = torch.cat([x, x_neg], dim=0)
        timestep_extended = torch.cat([timestep, t_neg], dim=0)
        
        # 3. Expand y (pooled text output) if present
        if y is not None:
            if nag_negative_y is not None:
                y_neg = nag_negative_y
                if nag_bsz > y_neg.shape[0]:
                    repeat_times = (nag_bsz + y_neg.shape[0] - 1) // y_neg.shape[0]
                    y_neg = y_neg.repeat(repeat_times, *[1]*(y_neg.ndim-1))[:nag_bsz]
                else:
                    y_neg = y_neg[:nag_bsz]
            else:
                if nag_bsz > pos_bsz:
                    repeat_times = (nag_bsz + pos_bsz - 1) // pos_bsz
                    y_neg = y.repeat(repeat_times, *[1]*(y.ndim-1))[:nag_bsz]
                else:
                    y_neg = y[:nag_bsz]
            y_extended = torch.cat([y, y_neg], dim=0)
        else:
            y_extended = None

        # 4. Expand any tensors in kwargs that match the original batch size (e.g. for negpip masks)
        def expand_tensors_in_dict(d, is_root=False):
            new_d = {}
            for k, v in d.items():
                # Skip expanding our own appended NAG parameters
                if is_root and k in["nag_negative_context", "nag_negative_y", "nag_sigma_start", "nag_sigma_end"]:
                    new_d[k] = v
                    continue
                
                # If we spot a batched tensor, duplicate the last part of it for the NAG batch
                if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == pos_bsz:
                    if nag_bsz > pos_bsz:
                        repeat_times = (nag_bsz + pos_bsz - 1) // pos_bsz
                        v_neg = v.repeat(repeat_times, *[1]*(v.ndim-1))[:nag_bsz]
                    else:
                        v_neg = v[:nag_bsz]
                    new_d[k] = torch.cat([v, v_neg], dim=0)
                elif isinstance(v, dict):
                    new_d[k] = expand_tensors_in_dict(v, is_root=False)
                elif k == "cond_or_uncond" and isinstance(v, list) and len(v) == pos_bsz:
                    new_d[k] = v + [v[-1]] * nag_bsz
                else:
                    new_d[k] = v
            return new_d

        kwargs_extended = expand_tensors_in_dict(kwargs, is_root=True)

        # 5. Patch self_attn blocks temporarily to tell them the batch split
        if hasattr(self, 'blocks'):
            for block in self.blocks:
                if hasattr(block, 'self_attn'):
                    block.self_attn.origin_bsz = nag_bsz
            
        try:
            # We use forward_orig_anima which we backed up in set_nag()
            out = self.forward_orig_anima(
                x_extended, 
                timestep_extended, 
                context_extended, 
                y=y_extended, 
                **kwargs_extended
            )
        finally:
            # Clean up the origin_bsz state immediately
            if hasattr(self, 'blocks'):
                for block in self.blocks:
                    if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'origin_bsz'):
                        delattr(block.self_attn, 'origin_bsz')
                        
        # 6. Return only the positive (guided) batch
        return out[:pos_bsz]
    else:
        # Standard forward (no NAG)
        return self.forward_orig_anima(x, timestep, context, y=y, **kwargs)


class NAGAnimaSwitch(NAGSwitch):
    """
    Switcher for enabling/disabling NAG on Anima / Cosmos models.
    """
    def set_nag(self):
        # 1. Safely override the model's main forward
        if getattr(self.model, 'forward', None) is not None and not hasattr(self.model, 'forward_orig_anima'):
            self.model.forward_orig_anima = self.model.forward

        self.model.forward = MethodType(
            partial(
                forward_nag_anima,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_negative_y=self.nag_negative_cond[0][1].get("pooled_output") 
                    if self.nag_negative_cond[0][1].get("pooled_output") is not None else None,
                nag_sigma_start=self.nag_sigma_start,
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model,
        )
        
        # 2. Patch the self_attn module inside each block directly
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                if hasattr(block, 'self_attn'):
                    # Store parameters directly on self_attn to avoid PyTorch circular dependencies
                    block.self_attn.nag_scale = self.nag_scale
                    block.self_attn.nag_tau = self.nag_tau
                    block.self_attn.nag_alpha = self.nag_alpha
                    
                    # Wrap self_attn if not already wrapped
                    if not hasattr(block.self_attn, 'is_nag_wrapper'):
                        block.self_attn.original_forward = block.self_attn.forward
                        block.self_attn.forward = MethodType(nag_self_attn_forward, block.self_attn)
                        block.self_attn.is_nag_wrapper = True

    def set_origin(self):
        """Restore everything back to normal when sampling finishes"""
        # 1. Restore the main forward
        if hasattr(self.model, 'forward_orig_anima'):
            self.model.forward = self.model.forward_orig_anima
            delattr(self.model, 'forward_orig_anima')
            
        # 2. Restore all attention blocks
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                if hasattr(block, 'self_attn'):
                    # Restore original forward
                    if hasattr(block.self_attn, 'is_nag_wrapper'):
                        block.self_attn.forward = block.self_attn.original_forward
                        delattr(block.self_attn, 'original_forward')
                        delattr(block.self_attn, 'is_nag_wrapper')
                    
                    # Clean up our attached attributes
                    for attr in['nag_scale', 'nag_tau', 'nag_alpha', 'origin_bsz']:
                        if hasattr(block.self_attn, attr):
                            delattr(block.self_attn, attr)