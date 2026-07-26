# ComfyUI-GGUF
GGUF Quantization support for native ComfyUI models including the custom Q8_CR

> [!NOTE]  
> This is a fork of the original nodes, updated to support loading Ideogram 4 GGUFs and Krea 2 GGUFs. 
> To use it, clone `https://github.com/city96/ComfyUI-GGUF`and not the original repo.

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs. For further VRAM savings, a node to load a quantized version of the T5 text encoder is also included.

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on v0.27.0 or later.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and install the only dependency for inference (`pip install --upgrade gguf`)

```
git clone https://github.com/molbal/ComfyUI-GGUF
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/molbal/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

On MacOS sequoia, torch 2.4.1 seems to be required, as 2.6.X nightly versions cause a "M1 buffer is not large enough" error. See [this issue](https://github.com/city96/ComfyUI-GGUF/issues/107) for more information/workarounds.

## Usage

Simply use the GGUF Unet loader found under the `bootleg` category. Place the .gguf model files in your `ComfyUI/models/unet` folder.

LoRA loading is experimental but it should work with just the built-in LoRA loader node(s).

Pre-quantized models (🍴 icon on ones added by this fork):

- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)
- [stable-diffusion-3.5-large GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)
- [stable-diffusion-3.5-large-turbo GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)
- [Krea 2 (Both Turbo and Raw)](https://huggingface.co/molbal/krea2-gguf) 🍴
- [Ideogram 4](https://huggingface.co/molbal/ideogram-4-gguf) 🍴


> [!IMPORTANT]  
> Please note, that this fork does not support _K quants on diffusion models, only on text encoders. They may or may not load, but inference speed may be very slow. There may be other forks, or other custom nodes with better support for these quantization types.

Initial support for quantizing T5 has also been added recently, these can be used using the various `*CLIPLoader (gguf)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.

- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)
- [Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)🍴

See the instructions in the [tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools) folder for how to create your own quants.

## Native weight-only quantization

The converter supports two custom, global quantization modes for DiT/transformer
UNets:

- `Q8_CR` stores eligible 2-D Linear weights as per-row INT8 ConvRot. It uses
  ComfyUI's native `TensorWiseINT8Layout` path, so weights remain INT8 during
  inference.

Q8_CR keeps 1-D, small, and architecture-designated high-precision tensors
in FP32. Conv2d weights remain FP16 because these modes accelerate Linear
matrix multiplication only.

### Q8_CR platform support

Q8_CR does not require CUDA. It uses ComfyUI's `comfy_kitchen` layout backend:

- NVIDIA CUDA uses ComfyUI's optimized native INT8 backend when available.
- Linux and non-CUDA environments use the `comfy_kitchen` eager backend.
- CPU Q8_CR loading and inference are supported, but naturally slower than
  optimized CUDA inference.

Q4_PT is retired from conversion and loading until a performant W4A16 backend
is available. Its experimental implementation remains in the source for future
work, but it is no longer selectable or loadable.

Reconvert any `Q8_CR` GGUF created before ConvRot weights were marked as
pre-rotated. Older files load safely with native non-rotated INT8 instead.
