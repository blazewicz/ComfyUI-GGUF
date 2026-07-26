# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import json
import torch
import logging

import comfy.ops
import comfy.lora
import comfy.model_management
from .dequant import dequantize_tensor, is_quantized

def _valid_compute_dtype(dtype):
    return dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}

def _infer_compute_dtype(tensor_type, fallback=None):
    if _valid_compute_dtype(fallback):
        return fallback
    if tensor_type == gguf.GGMLQuantizationType.BF16:
        return torch.bfloat16
    if tensor_type == gguf.GGMLQuantizationType.F32:
        return torch.float32
    return torch.float16

def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split('.'):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True

# A bakcward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x
        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator # torch compile works, nightly before 2.8 release
    else:
        logging.info("ComfyUI-GGUF: Partial torch compile only, consider updating pytorch")
        return torch.compiler.disable

torch_compiler_disable = get_torch_compiler_disable_decorator()

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], compute_dtype=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
        self.compute_dtype = compute_dtype

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], compute_dtype=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        new.compute_dtype = getattr(self, "compute_dtype", None)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
                new_tensor,
                tensor_type = getattr(self, "tensor_type", None),
                tensor_shape = size,
                patches = getattr(self, "patches", []).copy(),
                compute_dtype = getattr(self, "compute_dtype", None),
        )

    @property
    def dtype(self):
        qtype = getattr(self, "tensor_type", None)
        if qtype in GGMLLayer.torch_compatible_tensor_types:
            return torch.Tensor(self).dtype
        return _infer_compute_dtype(qtype, getattr(self, "compute_dtype", None))

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k,v in state_dict.items():
            if k[prefix_len:] == "weight":
                if isinstance(v, GGMLTensor):
                    v.compute_dtype = self._ggml_compute_dtype(v, "weight")
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                if isinstance(v, GGMLTensor):
                    v.compute_dtype = self._ggml_compute_dtype(v, "bias")
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix+"weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _ggml_compute_dtype(self, tensor, param_name):
        if self.dequant_dtype is not None and self.dequant_dtype != "target":
            return self.dequant_dtype
        model_dtype = getattr(self, f"{param_name}_comfy_model_dtype", None)
        return _infer_compute_dtype(getattr(tensor, "tensor_type", None), model_dtype)

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype if self.dequant_dtype and self.dequant_dtype != "target" else torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

        return
        # This would return the dequantized state dict
        destination[prefix + "weight"] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + "bias"] = self.get_weight(self.bias)

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        # apply patches
        if len(patch_list) > 0:
            if self.patch_dtype is None:
                weight = comfy.lora.calculate_weight(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        weight = s.get_weight(s.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError

class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

def get_gguf_q8_ops(compute_dtype=torch.bfloat16, full_precision_mm=False):
    """
    Factory for an ops class that uses ComfyUI's native mixed_precision_ops INT8 path.
    Weights are kept as INT8 and matmul uses comfy_kitchen's TensorWiseINT8Layout.
    """
    BaseOps = comfy.ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=full_precision_mm,
    )

    class GGUFQ8Ops(BaseOps):
        class Linear(BaseOps.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                # Lazy init: don't allocate weight here; it will be loaded from state dict
                torch.nn.Module.__init__(self)
                self.factory_kwargs = {"device": device, "dtype": BaseOps._compute_dtype}
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)
                self._orig_shape = (out_features, in_features)
                self.tensor_class = None
                self._full_precision_mm = BaseOps._full_precision_mm
                self._full_precision_mm_config = False

            def _load_from_state_dict(self, *args):
                return comfy.ops._load_quantized_module(
                    self,
                    torch.nn.Module._load_from_state_dict.__get__(self, type(self)),
                    *args,
                    load_extra_params=True,
                )

    return GGUFQ8Ops


class GGUFQ4Ops(comfy.ops.manual_cast):
    """
    Ops class for PyTorch's CUDA INT4 matrix multiplication.

    GGUF stores a portable nibble-packed representation. The first forward pass
    for each layer converts it to PyTorch's hardware-specific packed layout on
    the active CUDA device, and subsequent forwards reuse that packed tensor.
    """
    class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.register_parameter("bias", None)
            self._orig_shape = (out_features, in_features)
            self._group_size = None
            self._pad = 0
            self._orig_in_features = in_features
            self._int4_cache = {}
            self._is_int4 = False

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = f"{prefix}weight"
            scale_key = f"{prefix}weight_scale"
            quant_key = f"{prefix}comfy_quant"

            weight = state_dict.pop(weight_key, None)
            scale = state_dict.pop(scale_key, None)
            quant_raw = state_dict.pop(quant_key, None)

            bias_key = f"{prefix}bias"
            bias = state_dict.pop(bias_key, None)
            if quant_raw is None:
                if weight is None:
                    missing_keys.append(weight_key)
                    return
                self.weight = torch.nn.Parameter(torch.Tensor(weight), requires_grad=False)
                if bias is not None:
                    self.bias = torch.nn.Parameter(torch.Tensor(bias), requires_grad=False)
                self._is_int4 = False
                return

            if weight is None or scale is None:
                raise RuntimeError(f"Missing INT4 tensors for {prefix}")

            quant_conf = json.loads(bytes(quant_raw.tolist()).decode("utf-8"))
            if quant_conf.get("format") != "int4_pytorch":
                raise ValueError(f"Unsupported INT4 format for {prefix}")
            self._group_size = quant_conf["group_size"]
            self._pad = quant_conf.get("pad", 0)
            orig_shape = tuple(quant_conf["orig_shape"])
            self._orig_in_features = orig_shape[1]
            self._orig_shape = orig_shape
            self.out_features = orig_shape[0]

            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
            self._is_int4 = True

            if bias is not None:
                self.bias = torch.nn.Parameter(bias, requires_grad=False)
            for key in (weight_key, scale_key, quant_key, bias_key):
                if key in missing_keys:
                    missing_keys.remove(key)

        @staticmethod
        def _inner_k_tiles():
            return 2

        def _packed_weight(self, device, dtype):
            if device.type != "cuda":
                raise RuntimeError("Q4_PT requires CUDA; no dequantization fallback is available.")
            if dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(f"Q4_PT requires FP16 or BF16 activations, got {dtype}.")

            cache_key = (device, dtype)
            cached = self._int4_cache.get(cache_key)
            if cached is not None:
                return cached

            serialized = self.weight.to(device=device, dtype=torch.uint8, non_blocking=True)
            packed = torch._convert_weight_to_int4pack(serialized, self._inner_k_tiles())
            scale_and_offset = self.weight_scale.to(device=device, dtype=dtype, non_blocking=True)
            self._int4_cache[cache_key] = (packed, scale_and_offset)
            return packed, scale_and_offset

        def forward(self, input):
            if self.weight is None:
                raise RuntimeError("Q4_PT weight was not loaded.")
            if not self._is_int4:
                weight = self.weight.to(device=input.device, dtype=input.dtype)
                bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, weight, bias)
            if self.weight_function or self.bias_function:
                raise RuntimeError("Q4_PT does not support weight patches or LoRAs without dequantization.")
            input_shape = input.shape
            input_2d = input.reshape(-1, input_shape[-1])
            if self._pad:
                input_2d = torch.nn.functional.pad(input_2d, (0, self._pad))

            packed, scale_and_offset = self._packed_weight(input.device, input.dtype)
            output = torch._weight_int4pack_mm(
                input_2d, packed, self._group_size, scale_and_offset
            )
            if self.bias is not None:
                output = output + self.bias.to(device=input.device, dtype=input.dtype)
            return output.reshape(*input_shape[:-1], self.out_features)
