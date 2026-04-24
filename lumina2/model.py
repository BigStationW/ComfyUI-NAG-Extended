from functools import partial
from types import MethodType
import torch
import comfy.patcher_extension
from comfy.ldm.lumina.model import NextDiT, JointAttention
from ..utils import check_nag_activation, NAGSwitch, cat_context
from .attention import NAGJointAttention

class NAGNextDiT(NextDiT):
    """NAG-enabled NextDiT model for Lumina2/Z-Image"""

    def forward(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        nag_negative_context=None,
        nag_sigma_start=14.7,
        nag_sigma_end=0.,
        **kwargs,
    ):
        transformer_options = kwargs.get("transformer_options", {})
        apply_nag = check_nag_activation(transformer_options, nag_sigma_start, nag_sigma_end)

        modules_patched = list()
        
        # Capture original batch size for post-processing restoration
        original_batch_size = x.shape[0]

        if apply_nag:
            pos_bsz = original_batch_size
            nag_bsz = nag_negative_context.shape[0]

            def expand_tensors_in_dict(d, is_root=False):
                if not isinstance(d, dict): return d
                new_d = {}
                for k, v in d.items():
                    if is_root and k in ["nag_negative_context", "nag_sigma_start", "nag_sigma_end"]:
                        new_d[k] = v
                        continue
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

            kwargs = expand_tensors_in_dict(kwargs, is_root=True)
            transformer_options = kwargs.get("transformer_options", {})
            
            # 1. Calculate Image Token Count for the Attention layer to use
            _, _, h, w = x.shape
            p = self.patch_size
            img_token_len = (h // p) * (w // p)

            # 2. Prepare Inputs
            # Merge context first (Positive + Negative/NAG)
            context = cat_context(context, nag_negative_context)
            
            # Calculate how many items we need to add to x to match the new context size
            target_batch_size = context.shape[0]
            needed_count = target_batch_size - original_batch_size

            # Extend latents and conditions if the context is larger than input (e.g. adding Negative path)
            if needed_count > 0:
                # We append the necessary amount of latents from the input to match the context.
                x_extra = x[:needed_count]
                x = torch.cat([x, x_extra], dim=0)

                if timesteps is not None:
                    t_extra = timesteps[:needed_count]
                    timesteps = torch.cat([timesteps, t_extra], dim=0)
                
                if attention_mask is not None:
                    # attention_mask might be 2D or 3D depending on implementation, slicing dim 0 is safe
                    m_extra = attention_mask[:needed_count]
                    attention_mask = torch.cat([attention_mask, m_extra], dim=0)

            # 3. Patch Modules safely
            for name, module in self.named_modules():
                # Check for 'qkv' attribute to ensure we only patch actual Attention layers
                if isinstance(module, JointAttention) and hasattr(module, "qkv"):
                    # CHANGE: Swizzle the class instead of MethodType binding
                    modules_patched.append((module, module.__class__))
                    module.__class__ = NAGJointAttention
                    module._nag_img_token_len = img_token_len

        try:
            # 4. Execute Model
            output = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self._forward,
                self,
                comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    transformer_options
                )
            ).execute(x, timesteps, context, num_tokens, attention_mask, **kwargs)

        finally:
            # 5. Restore Original Methods (Always run this, even if crash happens)
            if apply_nag:
                # CHANGE: Restore the original class type
                for mod, original_class in modules_patched:
                    mod.__class__ = original_class

                # Cleanup
                if hasattr(mod, '_nag_img_token_len'):
                    delattr(mod, '_nag_img_token_len')

        # 6. Post-process Output
        if apply_nag and isinstance(output, torch.Tensor):
            # Restore original batch size (discard the extra NAG guidance outputs)
            output = output[:original_batch_size]

        return output


class NAGNextDiTSwitch(NAGSwitch):
    """Switch to enable/disable NAG for NextDiT models"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        # Store original forward method for restoration
        self.original_forward = model.forward
        # Store original attribute states
        self.original_attributes = {}

    def set_nag(self):
        """Enable NAG mode on the model"""
        # Store the negative context reference
        self._nag_negative_context = self.nag_negative_cond[0][0]
        
        # Create a wrapper function instead of using partial to avoid capturing tensor refs
        def nag_forward_wrapper(model_self, *args, **kwargs):
            return NAGNextDiT.forward(
                model_self,
                *args,
                nag_negative_context=self._nag_negative_context,
                nag_sigma_start=self.nag_sigma_start,
                nag_sigma_end=self.nag_sigma_end,
                **kwargs
            )
        
        # Replace the model's forward method with NAG-enabled version
        self.model.forward = MethodType(nag_forward_wrapper, self.model)

        # Set NAG parameters on all JointAttention modules
        for name, module in self.model.named_modules():
            if isinstance(module, JointAttention):
                # Store original values if they exist
                self.original_attributes[id(module)] = {
                    'nag_scale': getattr(module, 'nag_scale', None),
                    'nag_tau': getattr(module, 'nag_tau', None),
                    'nag_alpha': getattr(module, 'nag_alpha', None),
                }
                
                # Set new NAG parameters
                module.nag_scale = self.nag_scale
                module.nag_tau = self.nag_tau
                module.nag_alpha = self.nag_alpha

    def set_origin(self):
        """Restore original model state and clean up"""
        super().set_origin()
        # Restore original forward method
        self.model.forward = self.original_forward
        
        # Restore or remove NAG attributes from JointAttention modules
        for name, module in self.model.named_modules():
            if isinstance(module, JointAttention):
                module_id = id(module)
                
                # Restore original attributes or remove them
                if module_id in self.original_attributes:
                    orig_attrs = self.original_attributes[module_id]
                    
                    for attr_name in ['nag_scale', 'nag_tau', 'nag_alpha']:
                        if orig_attrs[attr_name] is None:
                            # Remove attribute if it didn't exist originally
                            if hasattr(module, attr_name):
                                delattr(module, attr_name)
                        else:
                            # Restore original value
                            setattr(module, attr_name, orig_attrs[attr_name])
        
        # Clear stored references to help garbage collection
        if hasattr(self, '_nag_negative_context'):
            del self._nag_negative_context
        self.original_attributes.clear()
