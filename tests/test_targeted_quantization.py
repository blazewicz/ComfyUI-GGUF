import unittest
from collections import OrderedDict
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import gguf
import torch
import comfy.sd
from safetensors.torch import save_file

from tools.convert import (
    MEBIBYTE,
    ModelMinimaxH3,
    ModelTemplate,
    convert_file,
    detect_arch,
    plan_target_size_quantization,
    quantize_int8_convrot,
    resolve_quantization_device,
)


def load_gguf_loader():
    loader_path = Path(__file__).parents[1] / "loader.py"
    package_name = "comfyui_gguf_test"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.loader",
        loader_path,
        submodule_search_locations=[str(loader_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[package_name] = module
    sys.modules[f"{package_name}.loader"] = module
    spec.loader.exec_module(module)
    return module


class TargetSizeQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.model_arch = ModelTemplate()
        self.state_dict = OrderedDict(
            (f"blocks.{index}.weight", torch.ones((4096, 32), dtype=torch.float32))
            for index in range(3)
        )
        self.state_dict["normalization.weight"] = torch.ones((4096,), dtype=torch.float32)

    def test_reduces_center_core_layers_before_outer_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.4
        )

        self.assertEqual(plan["blocks.1.weight"], gguf.GGMLQuantizationType.Q5_0)
        self.assertEqual(plan["blocks.0.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(plan["blocks.2.weight"], gguf.GGMLQuantizationType.I8)
        self.assertLessEqual(selected_size, int(0.4 * MEBIBYTE))

    def test_reduces_one_dimensional_weights_only_after_all_core_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.222
        )

        for index in range(3):
            self.assertEqual(plan[f"blocks.{index}.weight"], gguf.GGMLQuantizationType.Q4_0)
        self.assertEqual(plan["normalization.weight"], gguf.GGMLQuantizationType.BF16)
        self.assertLessEqual(selected_size, int(0.222 * MEBIBYTE))

    def test_supports_standard_q8_baseline(self):
        plan, _, _ = plan_target_size_quantization(
            self.state_dict,
            self.model_arch,
            2.0,
            target_size_q8_type="Q8_0",
        )

        for index in range(3):
            self.assertEqual(plan[f"blocks.{index}.weight"], gguf.GGMLQuantizationType.Q8_0)

    def test_reports_minimum_when_target_is_unsupported(self):
        with self.assertRaisesRegex(ValueError, "smallest supported TARGET_SIZE output"):
            plan_target_size_quantization(self.state_dict, self.model_arch, 0.1)


class Q8CRConversionDeviceTests(unittest.TestCase):
    def test_auto_uses_cpu_without_cuda(self):
        with mock.patch("tools.convert.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_quantization_device("auto").type, "cpu")

    def test_cuda_requires_available_device(self):
        with mock.patch("tools.convert.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires an available CUDA device"):
                resolve_quantization_device("cuda")

    def test_cpu_quantization_stays_on_cpu(self):
        qdata, scale, _, _ = quantize_int8_convrot(
            torch.arange(256, dtype=torch.float32).reshape(1, 256),
            device=torch.device("cpu"),
        )

        self.assertEqual(qdata.device.type, "cpu")
        self.assertEqual(scale.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_quantization_has_cpu_equivalent_decode_error(self):
        torch.manual_seed(0)
        weight = torch.randn((32, 256), dtype=torch.float32)
        cpu_qdata, cpu_scale, _, _ = quantize_int8_convrot(weight, device=torch.device("cpu"))
        cuda_qdata, cuda_scale, _, _ = quantize_int8_convrot(weight, device=torch.device("cuda"))

        cpu_decoded = cpu_qdata.to(torch.float32) * cpu_scale
        cuda_decoded = cuda_qdata.cpu().to(torch.float32) * cuda_scale.cpu()
        self.assertTrue(torch.allclose(cpu_decoded, cuda_decoded, atol=1e-5, rtol=0))

    def test_auto_device_serializes_q8_cr_layout(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.float32),
            "final_layer.video_out.weight": torch.ones((96, 32), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q8_CR.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="auto",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight_scale"],
            gguf.GGMLQuantizationType.F32,
        )


class Qwen3VLDetectionMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_uses_minimax_32b_detection_marker_for_5120_hidden_size(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.zeros(5120),
            "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
        }

        self.loader.inject_qwen3vl_detection_markers(state_dict)

        self.assertEqual(
            comfy.sd.detect_te_model(state_dict),
            comfy.sd.TEModel.QWEN3VL_32B,
        )
        self.assertIn("visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.merger.linear_fc2.weight", state_dict)
        self.assertEqual(
            state_dict["visual.deepstack_merger_list.0.norm.weight"].shape,
            (4608,),
        )

    def test_uses_model_prefixed_detection_markers_for_8b(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.zeros(4096),
        }

        self.loader.inject_qwen3vl_detection_markers(state_dict)

        self.assertIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertEqual(
            state_dict["model.visual.merger.linear_fc2.weight"].shape,
            (4096, 4608),
        )


class MinimaxH3DetectionTests(unittest.TestCase):
    def test_detects_native_minimax_h3_checkpoint_layout(self):
        checkpoint_keys = {
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
            "final_layer.video_out.weight",
        }

        model_arch = detect_arch(checkpoint_keys)

        self.assertIsInstance(model_arch, ModelMinimaxH3)
        self.assertEqual(model_arch.arch, "minimax_h3")

    def test_keeps_adaln_curve_table_in_full_precision(self):
        model_arch = ModelMinimaxH3()

        self.assertIn("adaln_t_table", model_arch.keys_hiprec)

    def test_converts_to_minimax_h3_gguf_with_full_precision_adaln_table(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.float32),
            "final_layer.video_out.weight": torch.ones((96, 32), dtype=torch.float32),
            "adaln_t_table": torch.ones((32, 32), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q8_0.gguf"
            save_file(state_dict, str(source_path))

            converted_path, model_arch = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_0",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            architecture = reader.get_field("general.architecture")
            architecture_name = str(architecture.parts[architecture.data[-1]], "utf-8")
            del architecture
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(model_arch.arch, "minimax_h3")
        self.assertEqual(architecture_name, "minimax_h3")
        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.Q8_0,
        )
        self.assertEqual(
            tensor_types["adaln_t_table"],
            gguf.GGMLQuantizationType.F32,
        )


if __name__ == "__main__":
    unittest.main()
