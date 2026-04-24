# model.py

from functools import partial
from types import MethodType
from typing import Callable
import inspect
import torch
from torch import Tensor
from einops import rearrange

from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.model import Flux

from .layers import NAGKleinDoubleStreamBlock, NAGKleinSingleStreamBlock
from ..utils import cat_context, check_nag_activation, NAGSwitch


class NAGKlein(Flux):
    """
    NAG-enabled Klein model.
    """
    
    def forward_orig_klein(
            self,
            img: Tensor,
            img_ids: Tensor,
            txt: Tensor,
            txt_ids: Tensor,
            txt_ids_negative: Tensor,
            timesteps: Tensor,
            y: Tensor,
            guidance: Tensor = None,
            control=None,
            transformer_options={},
            attn_mask: Tensor = None,
            context_pad_len: int = 0,
            nag_pad_len: int = 0,
    ) -> Tensor:
        """
        Klein's forward with NAG support.
        """
        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})
        
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img_bsz = img.shape[0]
        txt_bsz = txt.shape[0]
        origin_bsz = txt_bsz - img_bsz
        
        # Process inputs
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        
        if self.params.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec_extended = torch.cat((vec, vec[-origin_bsz:]), dim=0)
        
        if hasattr(self, 'vector_in') and self.vector_in is not None:
            if y is None:
                y = torch.zeros((img_bsz, self.params.vec_in_dim), device=img.device, dtype=img.dtype)
            y_extended = torch.cat((y, y[-origin_bsz:]), dim=0)
            vec_extended = vec_extended + self.vector_in(y_extended[:, :self.params.vec_in_dim])
        
        if hasattr(self, 'txt_norm') and self.txt_norm is not None:
            txt = self.txt_norm(txt)
        txt = self.txt_in(txt)

        # Positional embeddings
        if img_ids is not None:
            ids_positive = torch.cat((txt_ids, img_ids), dim=1)
            ids_negative = torch.cat((txt_ids_negative, img_ids[-origin_bsz:] if origin_bsz > 0 else img_ids), dim=1)
            pe = self.pe_embedder(ids_positive)
            pe_negative = self.pe_embedder(ids_negative)
        else:
            pe = None
            pe_negative = None

        vec_orig = vec_extended
        
        # Compute global modulation for double blocks
        # Returns tuple of (ModulationOut, ModulationOut) for attn and mlp paths
        img_mod = self.double_stream_modulation_img(vec_extended[:-origin_bsz] if origin_bsz > 0 else vec_extended)
        txt_mod = self.double_stream_modulation_txt(vec_extended)
        
        vec_double = (img_mod, txt_mod)
        
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        
        # Double blocks
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block.forward(
                        img=args["img"],
                        txt=args["txt"],
                        vec=args["vec"],
                        pe=args["pe"],
                        pe_negative=args.get("pe_negative"),
                        attn_mask=args.get("attn_mask"),
                        context_pad_len=args.get("context_pad_len", 0),
                        nag_pad_len=args.get("nag_pad_len", 0),
                        transformer_options=args.get("transformer_options")
                    )
                    return out

                out = blocks_replace[("double_block", i)](
                    {
                        "img": img,
                        "txt": txt,
                        "vec": vec_double,
                        "pe": pe,
                        "pe_negative": pe_negative,
                        "attn_mask": attn_mask,
                        "context_pad_len": context_pad_len,
                        "nag_pad_len": nag_pad_len,
                        "transformer_options": transformer_options
                    },
                    {"original_block": block_wrap}
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block.forward(
                    img=img,
                    txt=txt,
                    vec=vec_double,
                    pe=pe,
                    pe_negative=pe_negative,
                    attn_mask=attn_mask,
                    context_pad_len=context_pad_len,
                    nag_pad_len=nag_pad_len,
                    transformer_options=transformer_options
                )

            if control is not None:
                control_i = control.get("input")
                if control_i is not None and i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:, :add.shape[1]] += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        # Concatenate for single stream
        img = torch.cat((img, img[-origin_bsz:] if origin_bsz > 0 else img), dim=0)
        x = torch.cat((txt, img), dim=1)

        # Single stream modulation - returns (ModulationOut,) tuple or similar
        vec_single = self.single_stream_modulation(vec_orig)
        # vec_single is already in the right format (tuple of ModulationOut)

        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        
        # Single blocks
        for i, block in enumerate(self.single_blocks):
            transformer_options["block_index"] = i
            
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block.forward(
                        args["img"],
                        vec=args["vec"],
                        pe=args["pe"],
                        pe_negative=args.get("pe_negative"),
                        attn_mask=args.get("attn_mask"),
                        txt_length=args.get("txt_length"),
                        origin_bsz=args.get("origin_bsz"),
                        context_pad_len=args.get("context_pad_len", 0),
                        nag_pad_len=args.get("nag_pad_len", 0),
                        transformer_options=args.get("transformer_options")
                    )
                    return out

                out = blocks_replace[("single_block", i)](
                    {
                        "img": x,
                        "vec": vec_single,
                        "pe": pe,
                        "pe_negative": pe_negative,
                        "attn_mask": attn_mask,
                        "txt_length": txt.shape[1],
                        "origin_bsz": origin_bsz,
                        "context_pad_len": context_pad_len,
                        "nag_pad_len": nag_pad_len,
                        "transformer_options": transformer_options
                    },
                    {"original_block": block_wrap}
                )
                x = out["img"]
            else:
                x = block.forward(
                    x,
                    vec=vec_single,
                    pe=pe,
                    pe_negative=pe_negative,
                    attn_mask=attn_mask,
                    txt_length=txt.shape[1],
                    origin_bsz=origin_bsz,
                    context_pad_len=context_pad_len,
                    nag_pad_len=nag_pad_len,
                    transformer_options=transformer_options
                )

            if control is not None:
                control_o = control.get("output")
                if control_o is not None and i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        x[:, txt.shape[1]:txt.shape[1] + add.shape[1], ...] += add

        if origin_bsz > 0:
            x = x[:-origin_bsz]
        x = x[:, txt.shape[1]:, ...]
        x = self.final_layer(x, vec_orig[:-origin_bsz] if origin_bsz > 0 else vec_orig)
        
        return x
    
    def forward(
            self,
            x,
            timestep,
            context,
            y=None,
            guidance=None,
            ref_latents=None,
            control=None,
            transformer_options={},
            nag_negative_context=None,
            nag_negative_y=None,
            nag_sigma_start=14.7,
            nag_sigma_end=0.,
            **kwargs,
    ):
        """
        Main forward with NAG support for Klein.
        """
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.patch_size

        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        
        img, img_ids = self.process_img(x, transformer_options=transformer_options)
        img_tokens = img.shape[1]
        
        # Handle reference latents
        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            ref_latents_method = kwargs.get("ref_latents_method", getattr(self.params, 'default_ref_method', 'index'))
            
            for ref in ref_latents:
                if ref_latents_method == "index":
                    index += getattr(self.params, 'ref_index_scale', 1)
                    h_offset = 0
                    w_offset = 0
                elif ref_latents_method == "uxo":
                    index = 0
                    h_offset = h_len * patch_size + h
                    w_offset = w_len * patch_size + w
                    h += ref.shape[-2]
                    w += ref.shape[-1]
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        apply_nag = check_nag_activation(transformer_options, nag_sigma_start, nag_sigma_end)
        
        if apply_nag and nag_negative_context is not None:
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
            
            origin_context_len = context.shape[1]
            nag_bsz = nag_negative_context.shape[0]
            nag_negative_context_len = nag_negative_context.shape[1]
            
            context = cat_context(context, nag_negative_context, trim_context=True)
            
            context_pad_len = context.shape[1] - origin_context_len
            nag_pad_len = context.shape[1] - nag_negative_context_len

            forward_orig_ = getattr(self, 'forward_orig', None)
            double_blocks_forward = []
            single_blocks_forward = []

            try:
                self.forward_orig = MethodType(NAGKlein.forward_orig_klein, self)

                for block in self.double_blocks:
                    double_blocks_forward.append(block.forward)
                    block.forward = MethodType(
                        partial(
                            NAGKleinDoubleStreamBlock.forward,
                            context_pad_len=context_pad_len,
                            nag_pad_len=nag_pad_len,
                        ),
                        block,
                    )

                for block in self.single_blocks:
                    single_blocks_forward.append(block.forward)
                    block.forward = MethodType(
                        partial(
                            NAGKleinSingleStreamBlock.forward,
                            txt_length=context.shape[1],
                            origin_bsz=nag_bsz,
                            context_pad_len=context_pad_len,
                            nag_pad_len=nag_pad_len,
                        ),
                        block,
                    )

                txt_ids = torch.zeros(
                    (bs, origin_context_len, len(self.params.axes_dim)), 
                    device=x.device, 
                    dtype=torch.float32
                )
                txt_ids_negative = torch.zeros(
                    (nag_bsz, nag_negative_context_len, len(self.params.axes_dim)), 
                    device=x.device, 
                    dtype=torch.float32
                )
                
                if hasattr(self.params, 'txt_ids_dims') and len(self.params.txt_ids_dims) > 0:
                    for i in self.params.txt_ids_dims:
                        txt_ids[:, :, i] = torch.linspace(
                            0, origin_context_len - 1, 
                            steps=origin_context_len, 
                            device=x.device, 
                            dtype=torch.float32
                        )
                        txt_ids_negative[:, :, i] = torch.linspace(
                            0, nag_negative_context_len - 1, 
                            steps=nag_negative_context_len, 
                            device=x.device, 
                            dtype=torch.float32
                        )
                
                out = self.forward_orig(
                    img=img,
                    img_ids=img_ids,
                    txt=context,
                    txt_ids=txt_ids,
                    txt_ids_negative=txt_ids_negative,
                    timesteps=timestep,
                    y=y,
                    guidance=guidance,
                    control=control,
                    transformer_options=transformer_options,
                    attn_mask=kwargs.get("attention_mask", None),
                    context_pad_len=context_pad_len,
                    nag_pad_len=nag_pad_len,
                )
                
            finally:
                if forward_orig_ is not None:
                    self.forward_orig = forward_orig_
                elif hasattr(self, 'forward_orig'):
                    delattr(self, 'forward_orig')
                    
                for i, block in enumerate(self.double_blocks):
                    if i < len(double_blocks_forward):
                        block.forward = double_blocks_forward[i]
                        
                for i, block in enumerate(self.single_blocks):
                    if i < len(single_blocks_forward):
                        block.forward = single_blocks_forward[i]

        else:
            txt_ids = torch.zeros(
                (bs, context.shape[1], len(self.params.axes_dim)), 
                device=x.device, 
                dtype=torch.float32
            )
            
            if hasattr(self.params, 'txt_ids_dims') and len(self.params.txt_ids_dims) > 0:
                for i in self.params.txt_ids_dims:
                    txt_ids[:, :, i] = torch.linspace(
                        0, context.shape[1] - 1, 
                        steps=context.shape[1], 
                        device=x.device, 
                        dtype=torch.float32
                    )
            
            sig = inspect.signature(Flux.forward_orig)
            pass_kwargs = {}
            if "attn_mask" in sig.parameters:
                pass_kwargs["attn_mask"] = kwargs.get("attention_mask", None)
            if "timestep_zero_index" in sig.parameters:
                pass_kwargs["timestep_zero_index"] = kwargs.get("timestep_zero_index", None)
            
            out = Flux.forward_orig(
                self, 
                img=img, 
                img_ids=img_ids, 
                txt=context, 
                txt_ids=txt_ids, 
                timesteps=timestep, 
                y=y, 
                guidance=guidance, 
                control=control, 
                transformer_options=transformer_options,
                **pass_kwargs
            )

        out = out[:, :img_tokens]
        return rearrange(
            out, 
            "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
            h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size
        )[:, :, :h_orig, :w_orig]


class NAGKleinSwitch(NAGSwitch):
    """
    Switcher for enabling/disabling NAG on Klein models.
    """
    def set_nag(self):
        self.model.forward = MethodType(
            partial(
                NAGKlein.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_negative_y=self.nag_negative_cond[0][1].get("pooled_output") 
                    if self.nag_negative_cond[0][1].get("pooled_output") is not None else None,
                nag_sigma_start=self.nag_sigma_start,
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model,
        )
        
        for block in self.model.double_blocks:
            block.nag_scale = self.nag_scale
            block.nag_tau = self.nag_tau
            block.nag_alpha = self.nag_alpha
            
        for block in self.model.single_blocks:
            block.nag_scale = self.nag_scale
            block.nag_tau = self.nag_tau
            block.nag_alpha = self.nag_alpha
