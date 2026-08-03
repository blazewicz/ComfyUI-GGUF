import unittest
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

import gguf
import torch
from safetensors.torch import save_file

from tools.convert import (
    MEBIBYTE,
    ModelMinimaxH3,
    ModelTemplate,
    convert_file,
    detect_arch,
    plan_target_size_quantization,
)


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
            self.state_dict, self.model_arch, 0.38
        )

        self.assertEqual(plan["blocks.1.weight"], gguf.GGMLQuantizationType.Q4_0)
        self.assertEqual(plan["blocks.0.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(plan["blocks.2.weight"], gguf.GGMLQuantizationType.I8)
        self.assertLessEqual(selected_size, int(0.38 * MEBIBYTE))

    def test_reduces_one_dimensional_weights_only_after_all_core_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.222
        )

        for index in range(3):
            self.assertEqual(plan[f"blocks.{index}.weight"], gguf.GGMLQuantizationType.Q4_0)
        self.assertEqual(plan["normalization.weight"], gguf.GGMLQuantizationType.BF16)
        self.assertLessEqual(selected_size, int(0.222 * MEBIBYTE))

    def test_reports_minimum_when_target_is_unsupported(self):
        with self.assertRaisesRegex(ValueError, "smallest supported TARGET_SIZE output"):
            plan_target_size_quantization(self.state_dict, self.model_arch, 0.1)


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
