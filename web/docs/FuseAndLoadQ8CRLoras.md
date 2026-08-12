# Fuse LoRAs & Load

Fuses one or more LoRAs into a selected diffusion model, writes a
content-addressed Q8_CR GGUF cache, and returns the generated ComfyUI `MODEL`.

## Inputs

- **source_model**: A safetensors checkpoint or an FP16/BF16/F32 or standard
  GGML diffusion-model GGUF.
- **quantization_device**: `auto` uses CUDA when available. Select `cuda` to
  require it.
- **LoRA rows**: Click **Add Lora**, choose an adapter, then set its strength.
  Each row can be toggled independently; **Toggle All** changes every row.

## Outputs

- **model**: The cached Q8_CR model for use by samplers and downstream nodes.
- **gguf_path**: The generated cache file.
- **cache_info**: Whether the requested combination was a cache hit or miss.

## Cache behavior

Files are stored under `ComfyUI/models/diffusion_models/fused_cache`. A cache
entry is keyed by the source model content, enabled LoRAs, strengths, and
quantization device. The source model is loaded only to create a cache miss and
is then released before the cached model is returned.

Q8_CR source GGUFs cannot be used as inputs because their ConvRot weights cannot
be safely restored before applying a LoRA delta.

## Attribution

The compact LoRA row UI is adapted from
[rgthree-comfy Power Lora Loader](https://github.com/rgthree/rgthree-comfy),
Copyright (c) 2023 Regis Gaughan, III (rgthree), under the MIT License.
