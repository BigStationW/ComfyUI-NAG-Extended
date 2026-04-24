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
            nag_sigma_start=14.7,
            nag_sigma_end=0.,

            **kwargs,
    ):
        apply_nag = check_nag_activation(transformer_options, nag_sigma_start, nag_sigma_end)
        if apply_nag:
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
                nag_sigma_start=self.nag_sigma_start,
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model
        )
        for name, module in self.model.named_modules():
            if "attn2" in name and isinstance(module, CrossAttention):
                module.nag_scale = self.nag_scale
                module.nag_tau = self.nag_tau
                module.nag_alpha = self.nag_alpha
