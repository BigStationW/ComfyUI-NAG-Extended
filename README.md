# ComfyUI-NAG-Extended

**This branch supports Flux 2 Klein and Anima**

[Klein 9b](https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced/blob/main/workflow/workflow_Flux2_Klein_9b_clip_text_encode_NAG.json) - [Klein 4b](https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced/blob/main/workflow/workflow_Flux2_Klein_4b_clip_text_encode_NAG.json) - [Anima](https://github.com/BigStationW/ComfyUI-NAG-Extended/blob/main/workflows/NAG-Anima-ComfyUI-Workflow.json)

## Installation

Navigate to the **ComfyUI/custom_nodes** folder, [open cmd](https://www.youtube.com/watch?v=bgSSJQolR0E&t=47s) and run:

```bash
git clone https://github.com/BigStationW/ComfyUI-NAG-Extended
```
Restart ComfyUI after installation.

## Intro

Implementation of [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models](https://chendaryen.github.io/NAG.github.io/) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

NAG restores effective negative prompting in few-step diffusion models, and complements CFG in multi-step sampling for improved quality and control.

Paper: https://arxiv.org/abs/2505.21179

Code: https://github.com/ChenDarYen/Normalized-Attention-Guidance

Wan2.1 Demo: https://huggingface.co/spaces/ChenDY/NAG_wan2-1-fast

LTX Video Demo: https://huggingface.co/spaces/ChenDY/NAG_ltx-video-distilled

Flux-Dev Demo: https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-dev

![comfyui-nag](workflow.png?cache=20250628)

## Nodes

- `KSamplerWithNAG`, `KSamplerWithNAG (Advanced)`, `SamplerCustomWithNAG`
- `BasicGuider`, `NAGCFGGuider`, `NAGCFGGuiderAdvanced`

## Usage

To use NAG, simply replace
- `KSampler` with `KSamplerWithNAG`.
- `KSamplerWithNAG (Advanced)` with `KSampler (Advanced)`.
- `SamplerCustomWithNAG` with `SamplerCustom`.
- `NAGGuider` with `BasicGuider`.
- `CFGGuider` with `NAGCFGGuider`.

We currently support `Flux`, `Flux Kontext`, `Wan`, `Vace Wan`, `Hunyuan Video`, `Choroma`, `SD3.5`, `SDXL` and `SD`.

Example workflows are available in the `./workflows` directory!

## Key Inputs

When working with a new model, it's recommended to first find a good combination of `nag_tau` and `nag_alpha`, which ensures that the negative guidance is effective without introducing artifacts.

Once you're satisfied, keep `nag_tau` and `nag_alpha` fixed and tune only `nag_scale` in most cases to control the strength of guidance.

Using `nag_sigma_end` to reduce computation without much quality drop.

For flow-based models like `Flux`, `nag_sigma_end = 0.75` achieves near-identical results with significantly improved speed. For diffusion-based `SDXL`, a good default is `nag_sigma_end = 4`.

- `nag_scale`: The scale for attention feature extrapolation. Higher values result in stronger negative guidance.
- `nag_tau`: The normalisation threshold. Higher values result in stronger negative guidance.
- `nag_alpha`: Blending factor between original and extrapolated attention. Higher values result in stronger negative guidance.
- `nag_sigma_end`: NAG will be active only until `nag_sigma_end`.

### Rule of Thumb

- For image-reference tasks (e.g., Image2Video), use lower `nag_tau` and `nag_alpha` to preserve the reference content more faithfully.
- For models that require more sampling steps and higher CFG, also prefer lower `nag_tau` and `nag_alpha`.
- For few-step models, you can use higher `nag_tau` and `nag_alpha` to have stronger negative guidance.
