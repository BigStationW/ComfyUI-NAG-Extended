from typing import Optional
from types import MethodType
from functools import partial

import torch
from einops import repeat

from comfy.ldm.flux.math import apply_rope
import comfy.model_management
import comfy.ldm.common_dit
from comfy.ldm.hidream.model import (
    HiDreamImageTransformer2DModel,
    HiDreamAttention,
    HiDreamImageTransformerBlock,
    attention,
)

from ..utils import nag, cat_context, check_nag_activation, NAGSwitch


class NAGHiDreamAttnProcessor_flashattn:
    """Attention processor used typically in processing the SD3-like self-attention projections."""
    def __init__(
            self,
            nag_scale: float = 1.0,
            nag_tau=2.5,
            nag_alpha=0.25,
            encoder_hidden_states_length: int = None,
            origin_batch_size: int = None,
    ):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.encoder_hidden_states_length = encoder_hidden_states_length
        self.origin_batch_size = origin_batch_size

    def __call__(
        self,
        attn,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        dtype = image_tokens.dtype
        batch_size = image_tokens.shape[0]
        origin_batch_size = self.origin_batch_size
        txt_batch_size = text_tokens.shape[0] if text_tokens is not None else batch_size
        if text_tokens is not None:
            assert txt_batch_size == batch_size + origin_batch_size

        query_i = attn.q_rms_norm(attn.to_q(image_tokens)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(image_tokens)).to(dtype=dtype)
        value_i = attn.to_v(image_tokens)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if image_tokens_masks is not None:
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(text_tokens)).to(dtype=dtype)
            key_t = attn.k_rms_norm_t(attn.to_k_t(text_tokens)).to(dtype=dtype)
            value_t = attn.to_v_t(text_tokens)

            query_t = query_t.view(txt_batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(txt_batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(txt_batch_size, -1, attn.heads, head_dim)

            query_i = torch.cat([query_i, query_i[-origin_batch_size:]], dim=0)
            key_i = torch.cat([key_i, key_i[-origin_batch_size:]], dim=0)
            value_i = torch.cat([value_i, value_i[-origin_batch_size:]], dim=0)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)

        else:
            num_text_tokens = self.encoder_hidden_states_length
            num_image_tokens = query_i.shape[1] - num_text_tokens
            query = query_i
            key = key_i
            value = value_i

        if query.shape[-1] == rope.shape[-3] * 2:
            query, key = apply_rope(query, key, rope)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, rope)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        query_negative, query = query[-origin_batch_size:], query[:-origin_batch_size]
        key_negative, key = key[-origin_batch_size:], key[:-origin_batch_size]
        value_negative, value = value[-origin_batch_size:], value[:-origin_batch_size]

        hidden_states = attention(query, key, value)
        hidden_states_negative = attention(query_negative, key_negative, value_negative)
        del query_negative, key_negative, value_negative, query, key, value

        hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
        hidden_states_i_negative, hidden_states_t_negative = torch.split(hidden_states_negative, [num_image_tokens, num_text_tokens], dim=1)

        # NAG
        hidden_states_i_positive = hidden_states_i[-origin_batch_size:]
        hidden_states_i_guidance = nag(hidden_states_i_positive, hidden_states_i_negative, self.nag_scale, self.nag_tau, self.nag_alpha)

        hidden_states_i[-origin_batch_size:] = hidden_states_i_guidance

        if not attn.single:
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            hidden_states_t_negative = attn.to_out_t(hidden_states_t_negative)
            hidden_states_t = torch.cat([hidden_states_t, hidden_states_t_negative], dim=0)
            return hidden_states_i, hidden_states_t

        else:
            hidden_states[-origin_batch_size:, :num_image_tokens] = hidden_states_i_guidance
            hidden_states_negative[:, :num_image_tokens] = hidden_states_i_guidance
            hidden_states = attn.to_out(hidden_states)
            hidden_states_negative = attn.to_out(hidden_states_negative)
            hidden_states = torch.cat([hidden_states, hidden_states_negative], dim=0)
            return hidden_states


class NAGHiDreamImageTransformerBlock(HiDreamImageTransformerBlock):
    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        transformer_options=None,
        **kwargs,
    ) -> torch.FloatTensor:
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(12, dim=-1)

        # 1. MM-Attention
        image_batch_size = image_tokens.shape[0]
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i[:image_batch_size]) + shift_msa_i[:image_batch_size]
        norm_text_tokens = self.norm1_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_msa_t) + shift_msa_t

        attn_output_i, attn_output_t = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            norm_text_tokens,
            rope = rope,
        )

        image_tokens = gate_msa_i[:image_batch_size] * attn_output_i + image_tokens
        text_tokens = gate_msa_t * attn_output_t + text_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i[:image_batch_size]) + shift_mlp_i[:image_batch_size]
        norm_text_tokens = self.norm3_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_mlp_t) + shift_mlp_t

        ff_output_i = gate_mlp_i[:image_batch_size] * self.ff_i(norm_image_tokens)
        ff_output_t = gate_mlp_t * self.ff_t(norm_text_tokens)
        image_tokens = ff_output_i + image_tokens
        text_tokens = ff_output_t + text_tokens
        return image_tokens, text_tokens


class NAGHiDreamImageTransformer2DModel(HiDreamImageTransformer2DModel):
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

    def forward_nag(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        encoder_hidden_states_llama3=None,
        image_cond=None,
        control = None,
        transformer_options = {},
    ) -> torch.Tensor:
        bs, c, h, w = x.shape
        if image_cond is not None:
            x = torch.cat([x, image_cond], dim=-1)
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        timesteps = t
        pooled_embeds = y
        T5_encoder_hidden_states = context

        img_sizes = None

        # spatial forward
        batch_size = hidden_states.shape[0]
        txt_batch_size = T5_encoder_hidden_states.shape[0]
        origin_batch_size = txt_batch_size - batch_size
        hidden_states_type = hidden_states.dtype

        # 0. time
        timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        timesteps = torch.cat([timesteps, timesteps[-origin_batch_size:]], dim=0)
        p_embedder = self.p_embedder(pooled_embeds)
        adaln_input = timesteps + p_embedder

        hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
        if image_tokens_masks is None:
            pH, pW = img_sizes[0]
            img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        hidden_states = self.x_embedder(hidden_states)

        # T5_encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states = encoder_hidden_states_llama3.movedim(1, 0)
        encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(txt_batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
            T5_encoder_hidden_states = T5_encoder_hidden_states.view(txt_batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(T5_encoder_hidden_states)

        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device, dtype=img_ids.dtype
        )
        ids = torch.cat((img_ids, txt_ids), dim=1)
        ids = torch.cat([ids, ids[-origin_batch_size:]], dim=0)
        rope = self.pe_embedder(ids)

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            hidden_states, initial_encoder_hidden_states = block(
                image_tokens = hidden_states,
                image_tokens_masks = image_tokens_masks,
                text_tokens = cur_encoder_hidden_states,
                adaln_input = adaln_input,
                rope = rope,
            )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, hidden_states[-origin_batch_size:]], dim=0)
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if image_tokens_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
            )
            image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            hidden_states = block(
                image_tokens=hidden_states,
                image_tokens_masks=image_tokens_masks,
                text_tokens=None,
                adaln_input=adaln_input,
                rope=rope,
            )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:-origin_batch_size]
        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, adaln_input[:-origin_batch_size])
        output = self.unpatchify(output, img_sizes)
        return -output[:, :, :h, :w]

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            encoder_hidden_states_llama3=None,
            image_cond=None,
            control=None,
            transformer_options={},

            nag_negative_y=None,
            nag_negative_context=None,
            nag_negative_encoder_hidden_states_llama=None,
            nag_sigma_start=14.7,
            nag_sigma_end=0.,
    ):
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
            
            y = torch.cat((y, nag_negative_y.to(y)), dim=0)
            context = cat_context(context, nag_negative_context)
            encoder_hidden_states_llama3 = cat_context(
                encoder_hidden_states_llama3, nag_negative_encoder_hidden_states_llama,
                trim_context=True,
                dim=2,
            )

            forward_nag_ = self.forward_nag
            blocks_forward = list()
            attn_processors = list()

            for module in self.modules():
                if isinstance(module, HiDreamImageTransformerBlock):
                    blocks_forward.append((module, module.forward))
                    module.forward = MethodType(NAGHiDreamImageTransformerBlock.forward, module)
                elif isinstance(module, HiDreamAttention):
                    attn_processors.append((module, module.processor))
                    module.processor = NAGHiDreamAttnProcessor_flashattn(
                        nag_scale=self.nag_scale,
                        nag_tau=self.nag_tau,
                        nag_alpha=self.nag_alpha,
                        encoder_hidden_states_length=context.shape[1],
                        origin_batch_size=nag_negative_context.shape[0],
                    )

            self.forward_nag = MethodType(NAGHiDreamImageTransformer2DModel.forward_nag, self)

        output = self.forward_nag(x, t, y, context, encoder_hidden_states_llama3, image_cond, control, transformer_options)

        if apply_nag:
            self.forward_nag = forward_nag_
            for block, forward_fn in blocks_forward:
                block.forward = forward_fn
            for module, processor in attn_processors:
                module.processor = processor

        return output


class NAGHiDreamImageTransformer2DModelSwitch(NAGSwitch):
    def set_nag(self):
        self.model.nag_scale = self.nag_scale
        self.model.nag_tau = self.nag_tau
        self.model.nag_alpha = self.nag_alpha
        self.model.forward_nag = self.model.forward
        self.model.forward = MethodType(
            partial(
                NAGHiDreamImageTransformer2DModel.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_negative_y=self.nag_negative_cond[0][1]["pooled_output"],
                nag_negative_encoder_hidden_states_llama=self.nag_negative_cond[0][1]["conditioning_llama3"],
                nag_sigma_start=self.nag_sigma_start,
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model
        )

    def set_origin(self):
        super().set_origin()
        if hasattr(self.model, "forward_nag"):
            del self.model.forward_nag
