import torch
from torch import Tensor

from comfy.ldm.flux.math import attention
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock, apply_mod

from ..utils import nag


class NAGDoubleStreamBlock(DoubleStreamBlock):
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

    def forward(
            self,
            img: Tensor,
            txt: Tensor,
            vec: Tensor,
            pe: Tensor,
            pe_negative: Tensor,
            attn_mask=None,
            modulation_dims_img=None,
            modulation_dims_txt=None,
            context_pad_len: int = 0,
            nag_pad_len: int = 0,
            transformer_options=None,
            **kwargs,
    ):
        origin_bsz = len(txt) - len(img)
        assert origin_bsz != 0

        img_mod1, img_mod2 = self.img_mod(vec[:-origin_bsz])
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3,
                                                                                                              1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3,
                                                                                                              1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        txt_q_negative, txt_q = txt_q[-origin_bsz:, :, nag_pad_len:], txt_q[:-origin_bsz, :, context_pad_len:]
        txt_k_negative, txt_k = txt_k[-origin_bsz:, :, nag_pad_len:], txt_k[:-origin_bsz, :, context_pad_len:]
        txt_v_negative, txt_v = txt_v[-origin_bsz:, :, nag_pad_len:], txt_v[:-origin_bsz, :, context_pad_len:]

        img_q_negative = img_q[-origin_bsz:]
        img_k_negative = img_k[-origin_bsz:]
        img_v_negative = img_v[-origin_bsz:]

        if self.flipped_img_txt:
            # run actual attention
            attn_negative = attention(
                torch.cat((img_q_negative, txt_q_negative), dim=2),
                torch.cat((img_k_negative, txt_k_negative), dim=2),
                torch.cat((img_v_negative, txt_v_negative), dim=2),
                pe=pe_negative, mask=attn_mask,
            )
            attn = attention(
                torch.cat((img_q, txt_q), dim=2),
                torch.cat((img_k, txt_k), dim=2),
                torch.cat((img_v, txt_v), dim=2),
                pe=pe, mask=attn_mask,
            )

            img_attn_negative, txt_attn_negative = attn_negative[:, :img.shape[1]], attn_negative[:, img.shape[1]:]
            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]
        else:
            # run actual attention
            attn_negative = attention(
                torch.cat((txt_q_negative, img_q_negative), dim=2),
                torch.cat((txt_k_negative, img_k_negative), dim=2),
                torch.cat((txt_v_negative, img_v_negative), dim=2),
                pe=pe_negative, mask=attn_mask,
            )
            attn = attention(
                torch.cat((txt_q, img_q), dim=2),
                torch.cat((txt_k, img_k), dim=2),
                torch.cat((txt_v, img_v), dim=2),
                pe=pe, mask=attn_mask,
            )

            txt_attn_negative, img_attn_negative = attn_negative[:, : txt.shape[1] - nag_pad_len], attn_negative[:, txt.shape[1] - nag_pad_len:]
            txt_attn, img_attn = attn[:, : txt.shape[1] - context_pad_len], attn[:, txt.shape[1] - context_pad_len:]

        # NAG
        img_attn_positive = img_attn[-origin_bsz:]
        img_attn_guidance = nag(img_attn_positive, img_attn_negative, self.nag_scale, self.nag_tau, self.nag_alpha)

        img_attn = torch.cat([img_attn[:-origin_bsz], img_attn_guidance], dim=0)

        # calculate the img bloks
        img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
        img = img + apply_mod(
            self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)),
            img_mod2.gate, None, modulation_dims_img)

        # calculate the txt bloks
        txt[:-origin_bsz, context_pad_len:].addcmul_(txt_mod1.gate[:-origin_bsz], self.txt_attn.proj(txt_attn))
        txt[-origin_bsz:, nag_pad_len:].addcmul_(txt_mod1.gate[-origin_bsz:], self.txt_attn.proj(txt_attn_negative))
        txt += apply_mod(
            self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)),
            txt_mod2.gate, None, modulation_dims_txt)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class NAGSingleStreamBlock(SingleStreamBlock):
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

    def forward(
            self,
            x: Tensor,
            vec: Tensor,
            pe: Tensor,
            pe_negative: Tensor,
            attn_mask=None,
            modulation_dims=None,

            txt_length: int = None,
            img_length: int = None,
            origin_bsz: int = None,
            context_pad_len: int = 0,
            nag_pad_len: int = 0,
            transformer_options=None,
            **kwargs,
    ) -> Tensor:
        mod= self.modulation(vec)[0]
        qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # NAG
        if txt_length is not None:
            def remove_pad_and_get_neg(feature, pad_dim=2):
                assert pad_dim in [1, 2]
                if pad_dim == 2:
                    feature_negative = feature[-origin_bsz:, :, nag_pad_len:]
                    feature = feature[:-origin_bsz, :, context_pad_len:]
                else:
                    feature_negative = feature[-origin_bsz:, nag_pad_len:]
                    feature = feature[:-origin_bsz, context_pad_len:]

                return feature_negative, feature

        else:
            def remove_pad_and_get_neg(feature, pad_dim=2):
                assert pad_dim in [1, 2]
                if pad_dim == 2:
                    feature_negative = torch.cat([feature[-origin_bsz:, :, :img_length], feature[-origin_bsz:, :, img_length + nag_pad_len:]], dim=2)
                    feature = torch.cat([feature[:-origin_bsz, :, :img_length], feature[:-origin_bsz, :, img_length + context_pad_len:]], dim=2)
                else:
                    feature_negative = torch.cat([feature[-origin_bsz:, :img_length], feature[-origin_bsz:, img_length + nag_pad_len:]], dim=1)
                    feature = torch.cat([feature[:-origin_bsz, :img_length], feature[:-origin_bsz, img_length + context_pad_len:]], dim=1)
                return feature_negative, feature

        q_negative, q = remove_pad_and_get_neg(q)
        k_negative, k = remove_pad_and_get_neg(k)
        v_negative, v = remove_pad_and_get_neg(v)

        # compute attention
        attn_negative = attention(q_negative, k_negative, v_negative, pe=pe_negative, mask=attn_mask)
        attn = attention(q, k, v, pe=pe, mask=attn_mask)

        if txt_length is not None:
            img_attn_negative = attn_negative[:, txt_length - nag_pad_len:]
            img_attn = attn[:, txt_length - context_pad_len:]
        else:
            img_attn_negative = attn_negative[:, :img_length]
            img_attn = attn[:, :img_length]

        img_attn_positive = img_attn[-origin_bsz:]
        img_attn_guidance = nag(img_attn_positive, img_attn_negative, self.nag_scale, self.nag_tau, self.nag_alpha)

        if txt_length is not None:
            attn_negative[:, txt_length - nag_pad_len:] = img_attn_guidance
            attn[-origin_bsz:, txt_length - context_pad_len:] = img_attn_guidance
        else:
            attn_negative[:, :img_length] = img_attn_guidance
            attn[-origin_bsz:, :img_length] = img_attn_guidance

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_negative, mlp = remove_pad_and_get_neg(mlp, pad_dim=1)
        output_negative = self.linear2(torch.cat((attn_negative, self.mlp_act(mlp_negative)), 2))
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        if txt_length is not None:
            x[:-origin_bsz, context_pad_len:] += apply_mod(output, mod.gate[:-origin_bsz], None, modulation_dims)
            x[-origin_bsz:, nag_pad_len:] += apply_mod(output_negative, mod.gate[-origin_bsz:], None, modulation_dims)
        else:
            x[:-origin_bsz, :img_length] += apply_mod(output[:, :img_length], mod.gate[:-origin_bsz], None, modulation_dims)
            x[:-origin_bsz, img_length + context_pad_len:] += apply_mod(output[:, img_length:], mod.gate[:-origin_bsz], None, modulation_dims)
            x[-origin_bsz:, :img_length] += apply_mod(output_negative[:, :img_length], mod.gate[-origin_bsz:], None, modulation_dims)
            x[-origin_bsz:, img_length + nag_pad_len:] += apply_mod(output_negative[:, img_length:], mod.gate[-origin_bsz:], None, modulation_dims)

        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x
