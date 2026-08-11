import hashlib
import json
import os

import gguf
import torch


_SUPPORTED_FACTOR_TYPES = {
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
    gguf.GGMLQuantizationType.Q8_0,
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


def load_gguf_lora(path):
    """Read a standard GGUF LoRA into ComfyUI's lora_A/lora_B dictionary form."""
    reader = gguf.GGUFReader(path)
    try:
        if _read_string_field(reader, "general.type") != "adapter":
            raise ValueError("GGUF file is not an adapter (general.type must be 'adapter').")
        if _read_string_field(reader, "adapter.type") != "lora":
            raise ValueError("GGUF adapter is not a LoRA (adapter.type must be 'lora').")

        alpha = _read_float_field(reader, "adapter.lora.alpha")
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

        if not pairs:
            raise ValueError("GGUF LoRA contains no factor tensors.")

        lora = {}
        targets = {}
        for base_name, pair in pairs.items():
            if set(pair) != {"a", "b"}:
                raise ValueError(f"GGUF LoRA target {base_name!r} is missing one factor.")
            down, up = pair["a"], pair["b"]
            if down.ndim != 2 or up.ndim != 2:
                raise ValueError(
                    f"GGUF LoRA target {base_name!r} is not 2-D; convolutional and "
                    "other adapter layouts are not supported."
                )
            if down.shape[0] != up.shape[1]:
                raise ValueError(
                    f"GGUF LoRA target {base_name!r} has incompatible factor shapes "
                    f"{tuple(down.shape)} and {tuple(up.shape)}."
                )
            target_name = base_name[:-len(".weight")] if base_name.endswith(".weight") else base_name
            lora[f"{target_name}.lora_A.weight"] = down
            lora[f"{target_name}.lora_B.weight"] = up
            if alpha is not None:
                lora[f"{target_name}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
            targets[target_name] = {
                "base_name": base_name,
                "down": down,
                "up": up,
                "alpha": alpha,
            }

        metadata = {
            "path": os.path.abspath(path),
            "architecture": _read_string_field(reader, "general.architecture", required=False),
            "alpha": alpha,
            "target_count": len(targets),
        }
        return lora, targets, metadata
    finally:
        reader.tensors.clear()
        reader.fields.clear()
        reader.data._mmap.close()


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
        if source.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"Fusion target {state_key!r} has unsupported dtype {source.dtype}. "
                "Fuse from an FP16, BF16, or FP32 checkpoint."
            )
        down = target["down"].to(device=device, dtype=torch.float32)
        up = target["up"].to(device=device, dtype=torch.float32)
        alpha = target["alpha"] if target["alpha"] is not None else down.shape[0]
        fused = source.to(device=device, dtype=torch.float32)
        fused.add_(up.matmul(down), alpha=strength * alpha / down.shape[0])
        state_dict[state_key] = fused.to(device="cpu", dtype=source.dtype)
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
