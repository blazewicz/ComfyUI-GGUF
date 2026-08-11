# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
import logging
import inspect
import collections
import os
import json
import tempfile

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import comfy.memory_management
import folder_paths

from .ops import GGMLTensor, GGMLOps, get_gguf_q8_ops, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader, gguf_tensor_count
from .dequant import dequantize_tensor, is_quantized, is_torch_compatible
from .tools.convert import (
    DEFAULT_TARGET_SIZE_Q8_TYPE,
    QUANT_TYPE_MAP,
    QUANTIZATION_DEVICE_OPTIONS,
    TARGET_SIZE_Q8_TYPES,
    TARGET_SIZE_QUANT_TYPE,
    convert_state_dict,
    convert_file,
    load_safetensors_metadata,
    load_state_dict,
    resolve_quantization_device,
)
from .lora import cache_key, fuse_targets_into_state_dict, load_gguf_lora, load_lora

def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])
update_folder_names_and_paths("lora_gguf", ["loras"])
update_folder_names_and_paths("vae_gguf", ["vae"])


class GGUFLoadProgress:
    """Coordinate one ComfyUI progress bar across one or more model files."""

    def __init__(self, paths):
        self.path_totals = {
            path: gguf_tensor_count(path) if path.endswith(".gguf") else 1
            for path in paths
        }
        self.total = sum(self.path_totals.values())
        self.pbar = comfy.utils.ProgressBar(self.total)
        self.completed = 0

    def callback_for(self, path):
        offset = self.completed
        total = self.path_totals[path]

        def update(current, _loader_total):
            self.pbar.update_absolute(offset + current, self.total)

        return update

    def complete_file(self, path):
        self.completed += self.path_totals[path]
        self.pbar.update_absolute(self.completed, self.total)

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)


    def pin_weight_to_device(self, key):
        op_key = key.rsplit('.', 1)[0]
        if not self.mmap_released and op_key in self.named_modules_to_munmap:
            # TODO: possible to OOM, find better way to detach
            self.named_modules_to_munmap[op_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    mmap_released = False
    named_modules_to_munmap = {}

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        unet_path = folder_paths.get_full_path("unet", unet_name)
        progress = GGUFLoadProgress([unet_path])
        sd, extra = gguf_sd_loader(unet_path, progress_callback=progress.callback_for(unet_path))
        progress.complete_file(unet_path)

        mode = extra.get("gguf_quant_mode")
        if mode == "int8_convrot":
            # Use ComfyUI native INT8 path (weights stay INT8)
            ops = get_gguf_q8_ops(compute_dtype=torch.bfloat16)()
        elif mode == "int4_pytorch":
            raise RuntimeError(
                "Q4_PT is retired because PyTorch's Ampere INT4 kernel is not "
                "performance-competitive. Reconvert the model as Q8_CR."
            )
        else:
            ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model

        kwargs = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = extra.get("metadata", {})

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, **kwargs,
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)


class VAELoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae_gguf"),),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "bootleg"
    TITLE = "VAE Loader (GGUF)"

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        progress = GGUFLoadProgress([vae_path])
        sd, extra = gguf_sd_loader(
            vae_path,
            handle_prefix=None,
            progress_callback=progress.callback_for(vae_path),
        )
        progress.complete_file(vae_path)

        if extra["arch_str"] != "minimax_h3_vae":
            raise ValueError(
                "VAE Loader (GGUF) currently supports only MiniMax H3 video VAE GGUF files."
            )
        if extra.get("gguf_quant_mode") != "int8_convrot":
            raise ValueError(
                "MiniMax H3 VAE GGUF must use Q8_CR so decoder Linear weights stay on the native INT8 path."
            )

        for key, value in tuple(sd.items()):
            if isinstance(value, GGMLTensor):
                sd[key] = dequantize_tensor(value, dtype=torch.Tensor(value).dtype)

        operations = get_gguf_q8_ops(compute_dtype=torch.float16)()
        vae = comfy.sd.VAE(
            sd=sd,
            metadata=extra.get("metadata", {}),
            operations=operations,
            disable_dynamic=True,
        )
        vae.throw_exception_if_invalid()
        return (vae,)


class TargetedQuantizationGGUF:
    """Convert a source checkpoint to GGUF from a ComfyUI workflow."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Absolute path to a .safetensors, .ckpt, .pt, .bin, or .pth source model.",
                    },
                ),
                "destination_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Output .gguf path. Leave empty to derive it from the source path.",
                    },
                ),
                "quantization": (
                    [TARGET_SIZE_QUANT_TYPE, *QUANT_TYPE_MAP.keys()],
                    {
                        "default": TARGET_SIZE_QUANT_TYPE,
                        "tooltip": (
                            "TARGET_SIZE starts at the selected Q8 type, reduces central core matrices "
                            "to Q5_0 then Q4_0, then ordinary 1-D tensors to BF16 only when necessary."
                        ),
                    },
                ),
                "max_size_mb": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000000.0,
                        "step": 1.0,
                        "tooltip": "Maximum output size in MiB. Required only for TARGET_SIZE.",
                    },
                ),
                "target_size_q8_type": (
                    list(TARGET_SIZE_Q8_TYPES),
                    {
                        "default": DEFAULT_TARGET_SIZE_Q8_TYPE,
                        "tooltip": (
                            "TARGET_SIZE baseline: Q8_CR uses native INT8 ConvRot; "
                            "Q8_0 uses standard GGUF Q8 before layers are reduced to Q4_0."
                        ),
                    },
                ),
                "quantization_device": (
                    list(QUANTIZATION_DEVICE_OPTIONS),
                    {
                        "default": "auto",
                        "tooltip": (
                            "Q8_CR conversion device. auto uses CUDA when available and "
                            "falls back to CPU per matrix when VRAM is insufficient."
                        ),
                    },
                ),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora_paths": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Absolute .safetensors or .gguf LoRA paths, one per line or comma-separated.",
                    },
                ),
                "lora_strengths": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated merge strengths matching lora_paths; blank uses 1.0.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gguf_path", "quantization_info")
    FUNCTION = "quantize"
    CATEGORY = "bootleg/quantization"
    TITLE = "Targeted Quantization (GGUF)"

    def quantize(
        self,
        source_path,
        destination_path,
        quantization,
        max_size_mb,
        target_size_q8_type,
        quantization_device,
        overwrite,
        lora_paths="",
        lora_strengths="",
    ):
        source_path = os.path.abspath(os.path.expanduser(source_path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source model does not exist: {source_path}")

        if quantization == TARGET_SIZE_QUANT_TYPE and max_size_mb <= 0:
            raise ValueError("TARGET_SIZE requires max_size_mb greater than zero.")
        lora_paths = _parse_lora_paths(lora_paths) if lora_paths.strip() else []
        lora_strengths = _parse_strengths(lora_strengths, len(lora_paths)) if lora_paths else []

        progress = {"bar": None, "read_total": None, "total": None}

        def report_progress(stage, current, total):
            if stage == "read":
                if progress["bar"] is None:
                    progress["read_total"] = total
                    progress["total"] = total * 2 + 1
                    progress["bar"] = comfy.utils.ProgressBar(progress["total"])
                progress["bar"].update_absolute(current, progress["total"])
            elif stage == "quantize":
                if progress["bar"] is None:
                    progress["read_total"] = 0
                    progress["total"] = total + 1
                    progress["bar"] = comfy.utils.ProgressBar(progress["total"])
                progress["bar"].update_absolute(progress["read_total"] + current, progress["total"])

        output_path, _ = convert_file(
            source_path,
            dst_path=destination_path or None,
            interact=False,
            overwrite=overwrite,
            quant_type_name=None if quantization == TARGET_SIZE_QUANT_TYPE else quantization,
            max_size_mb=max_size_mb if quantization == TARGET_SIZE_QUANT_TYPE else None,
            target_size_q8_type=target_size_q8_type,
            quantization_device=quantization_device,
            progress_callback=report_progress,
            lora_paths=lora_paths,
            lora_strengths=lora_strengths,
        )
        if progress["bar"] is not None:
            progress["bar"].update_absolute(progress["total"], progress["total"])

        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        info = f"{quantization}: {output_size_mb:.2f} MiB written to {output_path}"
        return (output_path, info)


def _gguf_lora_path(lora_name):
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise FileNotFoundError(f"GGUF LoRA does not exist: {lora_name}")
    if not path.lower().endswith(".gguf"):
        raise ValueError(f"GGUF LoRA Import requires a .gguf file, got {lora_name!r}.")
    return path


def _gguf_lora_key_map(model, clip):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    return key_map


def _remap_gguf_lora_for_comfy(targets, key_map):
    lora = {}
    missing = []
    for target_name, target in targets.items():
        candidates = (
            target_name,
            f"diffusion_model.{target_name}",
            f"text_encoders.{target_name}",
            target_name.removeprefix("diffusion_model."),
            target_name.removeprefix("text_encoders."),
        )
        mapped_name = next((candidate for candidate in candidates if candidate in key_map), None)
        if mapped_name is None:
            missing.append(target["base_name"])
            continue
        lora[f"{mapped_name}.lora_A.weight"] = target["down"]
        lora[f"{mapped_name}.lora_B.weight"] = target["up"]
        if target["alpha"] is not None:
            lora[f"{mapped_name}.alpha"] = torch.tensor(
                target["alpha"], dtype=torch.float32
            )
    if missing:
        raise ValueError(
            "GGUF LoRA targets do not match the connected MODEL/CLIP: "
            + ", ".join(sorted(missing))
        )
    return lora


class GGUFLoraImport:
    """Load standard GGUF LoRA factors through ComfyUI's normal patch mechanism."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("lora_gguf"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
            "optional": {
                "clip": ("CLIP",),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "bootleg/LoRA"
    TITLE = "Load LoRA (GGUF)"

    def load_lora(self, model, lora_name, strength_model, clip=None, strength_clip=1.0):
        path = _gguf_lora_path(lora_name)
        _, targets, metadata = load_gguf_lora(path)
        key_map = _gguf_lora_key_map(model, clip)
        lora = _remap_gguf_lora_for_comfy(targets, key_map)
        return comfy.sd.load_lora_for_models(
            model,
            clip,
            lora,
            strength_model,
            strength_clip,
            lora_metadata={"gguf_lora": metadata},
        )


def _parse_lora_paths(lora_paths):
    paths = [
        os.path.abspath(os.path.expanduser(path.strip()))
        for path in lora_paths.replace(",", "\n").splitlines()
        if path.strip()
    ]
    if not paths:
        raise ValueError("Provide at least one LoRA path.")
    for path in paths:
        if not path.lower().endswith((".gguf", ".safetensors")):
            raise ValueError(f"LoRA fusion accepts only .gguf or .safetensors adapters, got {path!r}.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LoRA does not exist: {path}")
    return paths


def _parse_strengths(strengths, count):
    values = [value.strip() for value in strengths.replace("\n", ",").split(",") if value.strip()]
    if not values:
        return [1.0] * count
    if len(values) != count:
        raise ValueError(
            f"Expected {count} LoRA strength value(s), received {len(values)}."
        )
    try:
        return [float(value) for value in values]
    except ValueError as error:
        raise ValueError("LoRA strengths must be comma-separated numbers.") from error


class FuseGGUFLorasQ8CR:
    """Fuse fixed LoRA combinations into a content-addressed Q8_CR cache."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_path": (
                    "STRING",
                    {"default": "", "tooltip": "FP16, BF16, or FP32 diffusion checkpoint to fuse."},
                ),
                "lora_paths": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Absolute .safetensors or .gguf LoRA paths, one per line or comma-separated.",
                    },
                ),
                "strengths": (
                    "STRING",
                    {
                        "default": "1.0",
                        "tooltip": "Comma-separated strengths in the same order as lora_paths.",
                    },
                ),
                "cache_directory": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Leave empty to create gguf_lora_cache next to the source checkpoint.",
                    },
                ),
                "quantization_device": (
                    list(QUANTIZATION_DEVICE_OPTIONS),
                    {
                        "default": "auto",
                        "tooltip": "Fuses and quantizes on CUDA when available; cuda requires a CUDA device.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gguf_path", "cache_info")
    FUNCTION = "fuse"
    CATEGORY = "bootleg/LoRA"
    TITLE = "Fuse LoRAs (Q8_CR Cache)"

    def fuse(self, source_path, lora_paths, strengths, cache_directory, quantization_device):
        source_path = os.path.abspath(os.path.expanduser(source_path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source checkpoint does not exist: {source_path}")
        if source_path.lower().endswith(".gguf"):
            raise ValueError(
                "Fuse GGUF LoRAs requires an FP16, BF16, or FP32 checkpoint source. "
                "Do not fuse into an already quantized GGUF."
            )

        lora_paths = _parse_lora_paths(lora_paths)
        strengths = _parse_strengths(strengths, len(lora_paths))
        cache_directory = os.path.abspath(
            os.path.expanduser(cache_directory)
            if cache_directory.strip()
            else os.path.join(os.path.dirname(source_path), "gguf_lora_cache")
        )
        os.makedirs(cache_directory, exist_ok=True)

        cache_id, provenance = cache_key(
            source_path, lora_paths, strengths, quantization_device
        )
        output_path = os.path.join(cache_directory, f"lora-fused-{cache_id[:16]}-Q8_CR.gguf")
        metadata_path = f"{output_path}.json"
        if os.path.isfile(output_path) and os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                if json.load(metadata_file) == provenance:
                    return output_path, f"cache hit: {output_path}"

        device = resolve_quantization_device(quantization_device)
        if device.type != "cuda":
            logging.warning(
                "GGUF LoRA fusion is using CPU because CUDA is unavailable; "
                "set quantization_device to cuda to require GPU fusion."
            )
        state_dict = load_state_dict(source_path)
        for lora_path, strength in zip(lora_paths, strengths):
            _, targets, _ = load_lora(lora_path)
            fuse_targets_into_state_dict(state_dict, targets, strength, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_directory,
                prefix=f".{cache_id[:16]}-",
                suffix=".gguf",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
            convert_state_dict(
                state_dict,
                dst_path=temporary_path,
                source_path=source_path,
                source_metadata=load_safetensors_metadata(source_path),
                interact=False,
                overwrite=True,
                quant_type_name="Q8_CR",
                quantization_device=quantization_device,
            )
            os.replace(temporary_path, output_path)
            temporary_path = None
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(provenance, metadata_file, indent=2, sort_keys=True)
            return output_path, f"cache miss: fused and wrote {output_path}"
        finally:
            if temporary_path is not None and os.path.isfile(temporary_path):
                os.unlink(temporary_path)


def _require_dynamic_vram():
    if not comfy.memory_management.aimdo_enabled:
        raise RuntimeError(
            "Dynamic VRAM is not enabled in this ComfyUI installation. "
            "Start ComfyUI without --disable-dynamic-vram and use a build that supports DynamicVRAM."
        )


def _clone_as_dynamic_gguf_patcher(model_patcher):
    source_class = model_patcher.__class__
    model_patcher.__class__ = GGUFModelPatcherDynamic
    cloned = model_patcher.clone()
    model_patcher.__class__ = source_class
    return cloned


def _legacy_gguf_ops(extra):
    mode = extra.get("gguf_quant_mode")
    if mode == "int8_convrot":
        return get_gguf_q8_ops(compute_dtype=torch.bfloat16)()
    if mode == "int4_pytorch":
        raise RuntimeError(
            "Q4_PT is retired because PyTorch's Ampere INT4 kernel is not "
            "performance-competitive. Reconvert the model as Q8_CR."
        )
    return GGMLOps()


def _load_dynamic_gguf_unet(unet_path, disable_dynamic=False, progress=None):
    if not disable_dynamic:
        _require_dynamic_vram()
    progress = progress or GGUFLoadProgress([unet_path])
    sd, extra = gguf_sd_loader(
        unet_path,
        dynamic=not disable_dynamic,
        progress_callback=progress.callback_for(unet_path),
    )
    progress.complete_file(unet_path)

    kwargs = {}
    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get("metadata", {})

    # Target-size models combine native Q8_CR layers with standard GGML Q4_0
    # layers. Dynamic VRAM's default mixed-precision Linear cannot serialize a
    # bare GGML Q4_0 weight, so use the GGUF Q8 ops for both paths. Those ops
    # retain Q8_CR metadata and materialize only standard GGML layers as needed.
    model_options = {
        "custom_operations": _legacy_gguf_ops(extra),
    }
    model = comfy.sd.load_diffusion_model_state_dict(
        sd,
        model_options=model_options,
        disable_dynamic=disable_dynamic,
        **kwargs,
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
    model = GGUFModelPatcher.clone(model) if disable_dynamic else _clone_as_dynamic_gguf_patcher(model)
    model.cached_patcher_init = (_load_dynamic_gguf_unet, (unet_path,))
    return model


class GGUFModelPatcherDynamic(comfy.model_patcher.ModelPatcherDynamic):
    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        # GGML weights cannot be requantized after applying a LoRA patch.
        for _, module in self.model.named_modules():
            for param_key in ("weight", "bias"):
                attr = f"{param_key}_lowvram_function"
                lowvram_function = getattr(module, attr, None)
                if lowvram_function is not None:
                    setattr(module, attr, None)
                    functions = getattr(module, f"{param_key}_function", [])
                    functions.append(lowvram_function)
                    setattr(module, f"{param_key}_function", functions)

    def clone(self, disable_dynamic=False, model_override=None):
        if disable_dynamic:
            if model_override is None:
                fallback = self.cached_patcher_init[0](
                    *self.cached_patcher_init[1],
                    disable_dynamic=True,
                )
                model_override = fallback.get_clone_model_override()
            return GGUFModelPatcher.clone(self, model_override=model_override)
        return super().clone(disable_dynamic=disable_dynamic, model_override=model_override)


class UnetLoaderGGUFDynamicVRAM(UnetLoaderGGUF):
    TITLE = "Unet Loader (Dynamic VRAM)"

    def load_unet(self, unet_name, **kwargs):
        unet_path = folder_paths.get_full_path("unet", unet_name)
        return (_load_dynamic_gguf_unet(unet_path, progress=GGUFLoadProgress([unet_path])),)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }
    TITLE = "Unet Loader (GGUF/Advanced)"

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": base["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "bootleg"
    TITLE = "CLIPLoader (GGUF)"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        clip_data = []
        progress = GGUFLoadProgress(ckpt_paths)
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                sd = gguf_clip_loader(p, progress_callback=progress.callback_for(p))
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
                if "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                    raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
            clip_data.append(sd)
            progress.complete_file(p)
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type = clip_type,
            state_dicts = clip_data,
            model_options = {
                "custom_operations": GGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device()
            },
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher([clip_path], clip_type, self.load_data([clip_path])),)


def _load_dynamic_gguf_clip(clip_paths, clip_type, disable_dynamic=False, progress=None):
    if not disable_dynamic:
        _require_dynamic_vram()
    progress = progress or GGUFLoadProgress(clip_paths)
    clip_data = []
    for path in clip_paths:
        if path.endswith(".gguf"):
            clip_data.append(
                gguf_clip_loader(
                    path,
                    dynamic=not disable_dynamic,
                    progress_callback=progress.callback_for(path),
                )
            )
        else:
            clip_data.append(comfy.utils.load_torch_file(path, safe_load=True))
        progress.complete_file(path)

    model_options = {
        "initial_device": comfy.model_management.text_encoder_offload_device(),
    }
    if disable_dynamic:
        model_options["custom_operations"] = GGMLOps

    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=clip_type,
        state_dicts=clip_data,
        model_options=model_options,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        disable_dynamic=disable_dynamic,
    )
    clip.patcher = (
        GGUFModelPatcher.clone(clip.patcher)
        if disable_dynamic
        else _clone_as_dynamic_gguf_patcher(clip.patcher)
    )
    clip.patcher.cached_patcher_init = (_load_dynamic_gguf_clip_patcher, (clip_paths, clip_type))
    return clip


def _load_dynamic_gguf_clip_patcher(clip_paths, clip_type, disable_dynamic=False):
    return _load_dynamic_gguf_clip(
        clip_paths,
        clip_type,
        disable_dynamic=disable_dynamic,
    ).patcher


class CLIPLoaderGGUFDynamicVRAM(CLIPLoaderGGUF):
    TITLE = "CLIPLoader (Dynamic VRAM)"

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_dynamic_gguf_clip([clip_path], clip_type, progress=GGUFLoadProgress([clip_path])),)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": base["required"]["type"],
            }
        }

    TITLE = "DualCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_paths = (clip_path1, clip_path2)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class DualCLIPLoaderGGUFDynamicVRAM(DualCLIPLoaderGGUF):
    TITLE = "DualCLIPLoader (Dynamic VRAM)"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_paths = (
            folder_paths.get_full_path("clip", clip_name1),
            folder_paths.get_full_path("clip", clip_name2),
        )
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_dynamic_gguf_clip(clip_paths, clip_type),)

class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "clip_name3": file_options,
            }
        }

    TITLE = "TripleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_paths = (clip_path1, clip_path2, clip_path3)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class TripleCLIPLoaderGGUFDynamicVRAM(TripleCLIPLoaderGGUF):
    TITLE = "TripleCLIPLoader (Dynamic VRAM)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_paths = (
            folder_paths.get_full_path("clip", clip_name1),
            folder_paths.get_full_path("clip", clip_name2),
            folder_paths.get_full_path("clip", clip_name3),
        )
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_dynamic_gguf_clip(clip_paths, clip_type),)

class QuadrupleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
            "clip_name4": file_options,
        }
    }

    TITLE = "QuadrupleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_path4 = folder_paths.get_full_path("clip", clip_name4)
        clip_paths = (clip_path1, clip_path2, clip_path3, clip_path4)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class QuadrupleCLIPLoaderGGUFDynamicVRAM(QuadrupleCLIPLoaderGGUF):
    TITLE = "QuadrupleCLIPLoader (Dynamic VRAM)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        clip_paths = (
            folder_paths.get_full_path("clip", clip_name1),
            folder_paths.get_full_path("clip", clip_name2),
            folder_paths.get_full_path("clip", clip_name3),
            folder_paths.get_full_path("clip", clip_name4),
        )
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (_load_dynamic_gguf_clip(clip_paths, clip_type),)

NODE_CLASS_MAPPINGS = {
    "TargetedQuantizationGGUF": TargetedQuantizationGGUF,
    "GGUFLoraImport": GGUFLoraImport,
    "FuseGGUFLorasQ8CR": FuseGGUFLorasQ8CR,
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "VAELoaderGGUF": VAELoaderGGUF,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF,
    "QuadrupleCLIPLoaderGGUF": QuadrupleCLIPLoaderGGUF,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced,
    "UnetLoaderGGUFDynamicVRAM": UnetLoaderGGUFDynamicVRAM,
    "CLIPLoaderGGUFDynamicVRAM": CLIPLoaderGGUFDynamicVRAM,
    "DualCLIPLoaderGGUFDynamicVRAM": DualCLIPLoaderGGUFDynamicVRAM,
    "TripleCLIPLoaderGGUFDynamicVRAM": TripleCLIPLoaderGGUFDynamicVRAM,
    "QuadrupleCLIPLoaderGGUFDynamicVRAM": QuadrupleCLIPLoaderGGUFDynamicVRAM,
}
