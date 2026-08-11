import hashlib
import json
import os

import gguf
import torch
from safetensors import safe_open


_SUPPORTED_FACTOR_TYPES = {
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
    gguf.GGMLQuantizationType.Q8_0,
}
_FP8_SOURCE_TYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
}


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_string_field(reader, name, required=True):
    field = reader.get_field(name)
    if field is None:
        if required:
            raise ValueError(f"GGUF LoRA is missing required metadata {name!r}.")
        return None
    if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
        raise ValueError(f"GGUF LoRA metadata {name!r} must be a string.")
    return str(field.parts[field.data[-1]], encoding="utf-8")


def _read_float_field(reader, name):
    field = reader.get_field(name)
    if field is None:
        return None
    if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.FLOAT32:
        raise ValueError(f"GGUF LoRA metadata {name!r} must be a float32.")
    return float(field.parts[field.data[-1]].item())


def _tensor_to_float(tensor):
    if tensor.tensor_type not in _SUPPORTED_FACTOR_TYPES:
        raise ValueError(
            f"GGUF LoRA tensor {tensor.name!r} uses unsupported {tensor.tensor_type.name}. "
            "Only F32, F16, BF16, and Q8_0 factors are supported."
        )
    shape = tuple(int(value) for value in reversed(tensor.shape))
    data = torch.from_numpy(tensor.data.copy())
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        return data.view(torch.float32).reshape(shape)
    if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
        return data.view(torch.float16).reshape(shape)
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        return data.view(torch.bfloat16).reshape(shape)
    return torch.from_numpy(
        gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    ).reshape(shape).to(torch.float16)


def _build_targets(pairs, path, architecture=None, default_alpha=None):
    if not pairs:
        raise ValueError("LoRA contains no factor tensors.")

    lora = {}
    targets = {}
    for base_name, pair in pairs.items():
        if not {"a", "b"}.issubset(pair):
            raise ValueError(f"LoRA target {base_name!r} is missing one factor.")
        down, up = pair["a"], pair["b"]
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(
                f"LoRA target {base_name!r} is not 2-D; convolutional and other "
                "adapter layouts are not supported."
            )
        if down.shape[0] != up.shape[1]:
            raise ValueError(
                f"LoRA target {base_name!r} has incompatible factor shapes "
                f"{tuple(down.shape)} and {tuple(up.shape)}."
            )
        target_name = base_name.removesuffix(".weight")
        source_name = base_name if base_name.endswith(".weight") else f"{base_name}.weight"
        alpha = pair.get("alpha", default_alpha)
        lora[f"{target_name}.lora_A.weight"] = down
        lora[f"{target_name}.lora_B.weight"] = up
        if alpha is not None:
            lora[f"{target_name}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
        targets[target_name] = {
            "base_name": source_name,
            "down": down,
            "up": up,
            "alpha": alpha,
        }

    metadata = {
        "path": os.path.abspath(path),
        "architecture": architecture,
        "alpha": default_alpha,
        "target_count": len(targets),
    }
    return lora, targets, metadata


def load_gguf_lora(path):
    """Read a standard GGUF LoRA into ComfyUI's lora_A/lora_B dictionary form."""
    reader = gguf.GGUFReader(path)
    try:
        if _read_string_field(reader, "general.type") != "adapter":
            raise ValueError("GGUF file is not an adapter (general.type must be 'adapter').")
        if _read_string_field(reader, "adapter.type") != "lora":
            raise ValueError("GGUF adapter is not a LoRA (adapter.type must be 'lora').")

        pairs = {}
        for tensor in reader.tensors:
            if tensor.name.endswith(".lora_a"):
                base_name = tensor.name[:-len(".lora_a")]
                pairs.setdefault(base_name, {})["a"] = _tensor_to_float(tensor)
            elif tensor.name.endswith(".lora_b"):
                base_name = tensor.name[:-len(".lora_b")]
                pairs.setdefault(base_name, {})["b"] = _tensor_to_float(tensor)
            else:
                raise ValueError(
                    f"Unsupported GGUF LoRA tensor {tensor.name!r}; only paired "
                    "'.lora_a' and '.lora_b' factors are supported."
                )

        return _build_targets(
            pairs,
            path,
            architecture=_read_string_field(reader, "general.architecture", required=False),
            default_alpha=_read_float_field(reader, "adapter.lora.alpha"),
        )
    finally:
        reader.tensors.clear()
        reader.fields.clear()
        reader.data._mmap.close()


_SAFETENSORS_FACTOR_SUFFIXES = (
    (".lora_A.weight", "a"),
    (".lora_B.weight", "b"),
    (".lora_down.weight", "a"),
    (".lora_up.weight", "b"),
)


def load_safetensors_lora(path):
    """Load direct ComfyUI and Diffusers-style Linear LoRA factor names."""
    pairs = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for key in keys:
            match = next(
                ((suffix, factor) for suffix, factor in _SAFETENSORS_FACTOR_SUFFIXES if key.endswith(suffix)),
                None,
            )
            if match is None:
                continue
            suffix, factor = match
            base_name = key[:-len(suffix)]
            pairs.setdefault(base_name, {})[factor] = checkpoint.get_tensor(key)

        for base_name, pair in pairs.items():
            for alpha_key in (f"{base_name}.alpha", f"{base_name}.lora_alpha"):
                if alpha_key not in keys:
                    continue
                alpha = checkpoint.get_tensor(alpha_key)
                if alpha.numel() != 1:
                    raise ValueError(f"LoRA alpha {alpha_key!r} must be a scalar.")
                pair["alpha"] = float(alpha.item())
                break

    return _build_targets(pairs, path)


def load_lora(path):
    """Load a GGUF or safetensors LoRA suitable for offline fusion."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".gguf":
        return load_gguf_lora(path)
    if extension == ".safetensors":
        return load_safetensors_lora(path)
    raise ValueError(f"LoRA fusion accepts .gguf or .safetensors adapters, got {path!r}.")


def resolve_fusion_targets(state_dict, targets):
    resolved = {}
    missing = []
    for target_name, target in targets.items():
        candidates = (
            target["base_name"],
            f"{target_name}.weight",
            target_name,
            target["base_name"].removeprefix("diffusion_model."),
            f"{target_name.removeprefix('diffusion_model.')}.weight",
        )
        state_key = next((candidate for candidate in candidates if candidate in state_dict), None)
        if state_key is None:
            missing.append(target["base_name"])
            continue
        weight = state_dict[state_key]
        if weight.ndim != 2:
            raise ValueError(
                f"GGUF LoRA target {target['base_name']!r} resolves to {state_key!r}, "
                f"which is {weight.ndim}-D. Only 2-D Linear weights can be fused."
            )
        if tuple(weight.shape) != (target["up"].shape[0], target["down"].shape[1]):
            raise ValueError(
                f"GGUF LoRA target {target['base_name']!r} does not match {state_key!r}: "
                f"base shape {tuple(weight.shape)}, factors {tuple(target['up'].shape)} x "
                f"{tuple(target['down'].shape)}."
            )
        resolved[state_key] = target
    if missing:
        raise ValueError(
            "GGUF LoRA targets are not present in the source checkpoint: "
            + ", ".join(sorted(missing))
        )
    return resolved


def fuse_targets_into_state_dict(state_dict, targets, strength, device):
    """Apply one adapter's linear deltas in GPU/CPU FP32, returning the target count."""
    resolved = resolve_fusion_targets(state_dict, targets)
    for state_key, target in resolved.items():
        source = state_dict[state_key]
        source_dtype = source.dtype
        if source_dtype not in {torch.float16, torch.bfloat16, torch.float32, *_FP8_SOURCE_TYPES}:
            raise ValueError(
                f"Fusion target {state_key!r} has unsupported dtype {source.dtype}. "
                "Fuse from an FP16, BF16, FP32, or scaled FP8 checkpoint."
            )
        down = target["down"].to(device=device, dtype=torch.float32)
        up = target["up"].to(device=device, dtype=torch.float32)
        alpha = target["alpha"] if target["alpha"] is not None else down.shape[0]
        fused = source.to(device=device, dtype=torch.float32)
        if source_dtype in _FP8_SOURCE_TYPES:
            scale_key = f"{state_key}_scale"
            scale = state_dict.get(scale_key)
            if scale is not None:
                if scale.numel() != 1:
                    raise ValueError(f"FP8 scale tensor {scale_key!r} must be a scalar.")
                fused.mul_(scale.to(device=device, dtype=torch.float32))
        fused.add_(up.matmul(down), alpha=strength * alpha / down.shape[0])
        output_dtype = torch.float16 if source_dtype in _FP8_SOURCE_TYPES else source_dtype
        state_dict[state_key] = fused.to(device="cpu", dtype=output_dtype)
        del down, up, fused
    return len(resolved)


def cache_key(source_path, lora_paths, strengths, quantization_device):
    payload = {
        "source_sha256": file_sha256(source_path),
        "lora_sha256": [file_sha256(path) for path in lora_paths],
        "strengths": list(strengths),
        "quantization": "Q8_CR",
        "quantization_device": quantization_device,
        "format_version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload
