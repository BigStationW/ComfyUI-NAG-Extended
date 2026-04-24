from functools import partial
from types import MethodType

import torch
import comfy
from comfy.ldm.modules.attention import CrossAttention
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from .attention import NAGCrossAttention
from ..utils import cat_context, check_nag_activation, NAGSwitch


class NAGUNetModel(UNetModel):
    def forward(
            self,
            x,
            timesteps=None,
            context=None,
            y=None,
            control=None,
            transformer_options={},

            nag_negative_context=None,
            nag_sigma_end=0.,

            **kwargs,
    ):
        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)
        if apply_nag:
            pos_bsz = x.shape[0]
            nag_bsz = nag_negative_context.shape[0]

            def expand_tensors_in_dict(d, is_root=False):
                if not isinstance(d, dict): return d
                new_d = {}
                for k, v in d.items():
                    if is_root and k in ["nag_negative_context", "nag_sigma_end"]:
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

            transformer_options = expand_tensors_in_dict(transformer_options, is_root=True)
            kwargs = expand_tensors_in_dict(kwargs, is_root=True)
            
            context = cat_context(context, nag_negative_context)
            cross_attns_forward = list()
            for name, module in self.named_modules():
                if "attn2" in name and isinstance(module, CrossAttention):
                    cross_attns_forward.append((module, module.forward))
                    module.forward = MethodType(NAGCrossAttention.forward, module)

        output = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                                     transformer_options)
        ).execute(x, timesteps, context, y, control, transformer_options, **kwargs)

        if apply_nag:
            for mod, forward_fn in cross_attns_forward:
                mod.forward = forward_fn

        return output


class NAGUNetModelSwitch(NAGSwitch):
    def set_nag(self):
        self.model.forward = MethodType(
            partial(
                NAGUNetModel.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model
        )
        for name, module in self.model.named_modules():
            if "attn2" in name and isinstance(module, CrossAttention):
                module.nag_scale = self.nag_scale
                module.nag_tau = self.nag_tau
                module.nag_alpha = self.nag_alpha
