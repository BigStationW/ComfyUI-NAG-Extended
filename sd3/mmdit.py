from typing import Optional
from types import MethodType
from functools import partial
from typing import Callable

import torch
from einops import repeat
import comfy
from comfy.ldm.modules.diffusionmodules.mmdit import (
    OpenAISignatureMMDITWrapper,
    JointBlock,
    optimized_attention,
    default,
)

from ..utils import nag, cat_context, check_nag_activation, NAGSwitch


def _nag_block_mixing(
        context,
        x,
        context_block,
        x_block,
        c,
        nag_scale: float = 1.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.5,
):
    origin_bsz = len(context) - len(x)
    assert origin_bsz != 0

    context_qkv, context_intermediates = context_block.pre_attention(context, c)

    if x_block.x_block_self_attn:
        x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c[:-origin_bsz])
    else:
        x_qkv, x_intermediates = x_block.pre_attention(x, c[:-origin_bsz])

    o = []
    for t in range(3):
        o.append(torch.cat((
            context_qkv[t],
            torch.cat([x_qkv[t], x_qkv[t][-origin_bsz:]], dim=0),
        ),dim=1))
    qkv = tuple(o)

    attn = optimized_attention(
        qkv[0], qkv[1], qkv[2],
        heads=x_block.attn.num_heads,
    )
    context_attn, x_attn = (
        attn[:, : context_qkv[0].shape[1]],
        attn[:, context_qkv[0].shape[1] :],
    )

    # NAG
    x_attn_negative, x_attn_positive = x_attn[-origin_bsz:], x_attn[-origin_bsz * 2:-origin_bsz]
    x_attn_guidance = nag(x_attn_positive, x_attn_negative, nag_scale, nag_tau, nag_alpha)

    x_attn = torch.cat([x_attn[:-origin_bsz * 2], x_attn_guidance], dim=0)

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)

    else:
        context = None
    if x_block.x_block_self_attn:
        attn2 = optimized_attention(
                x_qkv2[0], x_qkv2[1], x_qkv2[2],
                heads=x_block.attn2.num_heads,
            )
        x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
    else:
        x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


def nag_block_mixing(*args, use_checkpoint=True, **kwargs):
    if use_checkpoint:
        return torch.utils.checkpoint.checkpoint(
            _nag_block_mixing, *args, use_reentrant=False, **kwargs
        )
    else:
        return _nag_block_mixing(*args, **kwargs)


class NAGJointBlock(JointBlock):
    def forward(self, *args, **kwargs):
        return nag_block_mixing(
            *args, context_block=self.context_block, x_block=self.x_block, **kwargs
        )


class NAGOpenAISignatureMMDITWrapper(OpenAISignatureMMDITWrapper):
    def __init__(
            self,
            *args,
            nag_scale: float = 1,
            nag_tau: float = 2.5,
            nag_alpha: float = 0.25,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def forward_core_with_concat(
        self,
        x: torch.Tensor,
        c_mod: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        control = None,
        transformer_options = {},
    ) -> torch.Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if self.register_length > 0:
            context = torch.cat(
                (
                    repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                    default(context, torch.Tensor([]).type_as(x)),
                ),
                1,
            )

        # context is B, L', D
        # x is B, L, D
        blocks_replace = patches_replace.get("dit", {})
        blocks = len(self.joint_blocks)
        for i in range(blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = self.joint_blocks[i](args["txt"], args["img"], c=args["vec"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": x,
                                                           "txt": context,
                                                           "vec": c_mod,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                context = out["txt"]
                x = out["img"]
            else:
                context, x = self.joint_blocks[i](
                    context,
                    x,
                    c=c_mod,
                    use_checkpoint=self.use_checkpoint,
                )
            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        x += add

        x = self.final_layer(x, c_mod[:len(x)])  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward_core_with_concat_with_wavespeed(
        self,
        x: torch.Tensor,
        c_mod: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        control = None,
        transformer_options = {},
        use_cache: Callable = None,
        apply_prev_hidden_states_residual: Callable = None,
        set_buffer: Callable = None,
    ) -> torch.Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if self.register_length > 0:
            context = torch.cat(
                (
                    repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                    default(context, torch.Tensor([]).type_as(x)),
                ),
                1,
            )

        # context is B, L', D
        # x is B, L, D
        blocks_replace = patches_replace.get("dit", {})
        joint_blocks = self.joint_blocks[0].transformer_blocks
        blocks = len(joint_blocks)

        original_x = x
        can_use_cache = False

        for i in range(blocks):
            if i == 1:
                torch._dynamo.graph_break()
                if can_use_cache:
                    del first_x_residual
                    x = apply_prev_hidden_states_residual(x)
                    break
                else:
                    set_buffer("first_hidden_states_residual", first_x_residual)
                    del first_x_residual

                    original_x = x

            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = joint_blocks[i](args["txt"], args["img"], c=args["vec"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": x,
                                                           "txt": context,
                                                           "vec": c_mod,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                context = out["txt"]
                x = out["img"]
            else:
                context, x = joint_blocks[i](
                    context,
                    x,
                    c=c_mod,
                    use_checkpoint=self.use_checkpoint,
                )
            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        x += add

            if i == 0:
                first_x_residual = x - original_x
                can_use_cache = use_cache(first_x_residual)
                del original_x

        if not can_use_cache:
            x = x.contiguous()
            x_residual = x - original_x
            set_buffer("hidden_states_residual", x_residual)
        torch._dynamo.graph_break()

        x = self.final_layer(x, c_mod[:len(x)])  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            control=None,
            transformer_options={},

            nag_negative_context=None,
            nag_negative_y=None,
            nag_sigma_start=14.7,
            nag_sigma_end=0.,

            **kwargs,
    ) -> torch.Tensor:
        apply_nag = check_nag_activation(transformer_options, nag_sigma_start, nag_sigma_end)
        if apply_nag:
            pos_bsz = x.shape[0]
            nag_bsz = nag_negative_context.shape[0]

            def expand_tensors_in_dict(d, is_root=False):
                if not isinstance(d, dict): return d
                new_d = {}
                for k, v in d.items():
                    if is_root and k in["nag_negative_context", "nag_negative_y", "nag_sigma_start", "nag_sigma_end", "nag_negative_encoder_hidden_states_llama"]:
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
            y = torch.cat((y, nag_negative_y.to(y)), dim=0)

            forward_core_with_concat_ = self.forward_core_with_concat
            joint_blocks_forward = list()

            joint_blocks = self.joint_blocks
            is_wavespeed = "CachedTransformerBlocks" in type(joint_blocks[0]).__name__
            if is_wavespeed:  # chengzeyi/Comfy-WaveSpeed
                cached_blocks = self.joint_blocks[0]
                joint_blocks = cached_blocks.transformer_blocks

            if is_wavespeed:
                get_can_use_cache = cached_blocks.forward.__globals__["get_can_use_cache"]
                set_buffer = cached_blocks.forward.__globals__["set_buffer"]
                apply_prev_hidden_states_residual = cached_blocks.forward.__globals__["apply_prev_hidden_states_residual"]

                def use_cache(first_hidden_states_residual):
                    return get_can_use_cache(
                        first_hidden_states_residual,
                        threshold=cached_blocks.residual_diff_threshold,
                        validation_function=cached_blocks.validate_can_use_cache_function,
                    )

                self.forward_core_with_concat = MethodType(
                    partial(
                        NAGOpenAISignatureMMDITWrapper.forward_core_with_concat_with_wavespeed,
                        use_cache=use_cache,
                        apply_prev_hidden_states_residual=apply_prev_hidden_states_residual,
                        set_buffer=set_buffer,
                    ),
                    self,
                )

            else:
                self.forward_core_with_concat = MethodType(NAGOpenAISignatureMMDITWrapper.forward_core_with_concat, self)

            for block in joint_blocks:
                joint_blocks_forward.append(block.forward)
                block.forward = MethodType(
                    partial(
                        NAGJointBlock.forward,
                        nag_scale=self.nag_scale,
                        nag_tau=self.nag_tau,
                        nag_alpha=self.nag_alpha,
                    ),
                    block,
                )

        if self.context_processor is not None:
            context = self.context_processor(context)

        hw = x.shape[-2:]
        x = self.x_embedder(x) + comfy.ops.cast_to_input(self.cropped_pos_embed(hw, device=x.device), x)
        c = self.t_embedder(timesteps, dtype=x.dtype)  # (N, D)

        if apply_nag:
            origin_bsz = len(context) - len(x)
            c = torch.cat((c, c[-origin_bsz:]), dim=0)

        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        if context is not None:
            context = self.context_embedder(context)

        x = self.forward_core_with_concat(x, c, context, control, transformer_options)

        if apply_nag:
            self.forward_core_with_concat = forward_core_with_concat_
            for block in joint_blocks:
                block.forward = joint_blocks_forward.pop(0)

        x = self.unpatchify(x, hw=hw)  # (N, out_channels, H, W)
        return x[:, :, :hw[-2], :hw[-1]]


class NAGOpenAISignatureMMDITWrapperSwitch(NAGSwitch):
    def set_nag(self):
        self.model.nag_scale = self.nag_scale
        self.model.nag_tau = self.nag_tau
        self.model.nag_alpha = self.nag_alpha
        self.model.forward = MethodType(
            partial(
                NAGOpenAISignatureMMDITWrapper.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_negative_y=self.nag_negative_cond[0][1]["pooled_output"],
                nag_sigma_start=self.nag_sigma_start,
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model
        )
