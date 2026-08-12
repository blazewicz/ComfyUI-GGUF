import unittest
import json
import gc
import os
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
    ModelMinimaxH3VAE,
    ModelTemplate,
    convert_file,
    detect_arch,
    plan_target_size_quantization,
    quantize_int8_convrot,
    resolve_quantization_device,
)
from dequant import dequantize_tensor
from lora import fuse_targets_into_state_dict, load_gguf_lora, load_lora


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


def load_nodes_module():
    nodes_path = Path(__file__).parents[1] / "nodes.py"
    package_name = "comfyui_gguf_nodes_test"
    import sys
    comfy_root = str(nodes_path.parents[2])
    if comfy_root in sys.path:
        sys.path.remove(comfy_root)
    sys.path.insert(0, comfy_root)
    existing_nodes = sys.modules.get("nodes")
    if existing_nodes is not None and Path(getattr(existing_nodes, "__file__", "")).resolve() == nodes_path:
        del sys.modules["nodes"]
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.nodes",
        nodes_path,
        submodule_search_locations=[str(nodes_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    sys.modules[f"{package_name}.nodes"] = module
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


class MiniMaxH3VAEConversionTests(unittest.TestCase):
    def test_detects_video_vae_from_distinctive_decoder_and_encoder_keys(self):
        state_dict = {
            "decoder.transformer_blocks.0.scale1": torch.ones(32),
            "decoder.x_embedder.weight": torch.ones((64, 32)),
            "encoder.down.5.block.0.conv1.weight": torch.ones((2, 2, 3, 3, 3)),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelMinimaxH3VAE)

    def test_q8_cr_keeps_conv3d_fp16_and_restores_its_shape(self):
        state_dict = {
            "decoder.transformer_blocks.0.scale1": torch.ones(32, dtype=torch.float16),
            "decoder.x_embedder.weight": torch.ones((64, 32), dtype=torch.float16),
            "encoder.down.5.block.0.conv1.weight": torch.ones(
                (2, 2, 3, 3, 3), dtype=torch.float16
            ),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3_vae.safetensors"
            output_path = Path(temp_dir) / "minimax_h3_vae-Q8_CR.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            loader = load_gguf_loader()
            loaded, extra = loader.gguf_sd_loader(converted_path, handle_prefix=None)
            conv3d = loaded["encoder.down.5.block.0.conv1.weight"]
            del conv3d.tensor_type
            materialized = dequantize_tensor(conv3d, dtype=torch.Tensor(conv3d).dtype)
            conv3d_shape = tuple(materialized.shape)
            del materialized
            del conv3d
            del loaded
            gc.collect()

        self.assertEqual(extra["arch_str"], "minimax_h3_vae")
        self.assertEqual(
            tensor_types["decoder.x_embedder.weight"], gguf.GGMLQuantizationType.I8
        )
        self.assertEqual(
            tensor_types["encoder.down.5.block.0.conv1.weight"],
            gguf.GGMLQuantizationType.F16,
        )
        self.assertEqual(
            conv3d_shape,
            (2, 2, 3, 3, 3),
        )


class GGUFLoraTests(unittest.TestCase):
    def _write_lora(self, path, down, up, alpha=4.0):
        writer = gguf.GGUFWriter(path=None, arch="minimax_h3")
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", alpha)
        writer.add_tensor(
            "blocks.0.attn.qkv_proj.weight.lora_a",
            down.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        writer.add_tensor(
            "blocks.0.attn.qkv_proj.weight.lora_b",
            up.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        writer.write_header_to_file(path=str(path))
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_imports_standard_gguf_factor_pair(self):
        down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        up = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.gguf"
            self._write_lora(path, down, up)
            lora, targets, metadata = load_gguf_lora(path)

        self.assertEqual(metadata["alpha"], 4.0)
        self.assertIn("blocks.0.attn.qkv_proj.lora_A.weight", lora)
        self.assertIn("blocks.0.attn.qkv_proj.lora_B.weight", lora)
        self.assertTrue(torch.equal(targets["blocks.0.attn.qkv_proj"]["down"], down))
        self.assertTrue(torch.equal(targets["blocks.0.attn.qkv_proj"]["up"], up))

    def test_fuses_lora_delta_in_selected_precision(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.zeros((3, 2), dtype=torch.float16)
        }
        targets = {
            "blocks.0.attn.qkv_proj": {
                "base_name": "blocks.0.attn.qkv_proj.weight",
                "down": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "up": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                "alpha": 4.0,
            }
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=0.5, device=torch.device("cpu")
        )

        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [4.0, 6.0]], dtype=torch.float16)
        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(state_dict["blocks.0.attn.qkv_proj.weight"], expected))


class FusedLoraCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node_module = load_nodes_module()

    def test_source_model_list_includes_standard_and_gguf_models(self):
        with mock.patch.object(
            self.node_module.folder_paths,
            "get_filename_list",
            side_effect=lambda folder: {
                "diffusion_models": ["base.safetensors"],
                "unet_gguf": ["base.gguf"],
            }[folder],
        ):
            self.assertEqual(
                self.node_module._get_fuse_source_names(),
                ["base.gguf", "base.safetensors"],
            )

    def test_cache_path_is_fixed_below_diffusion_models(self):
        with mock.patch.object(self.node_module.folder_paths, "models_dir", "/models"):
            self.assertEqual(
                self.node_module._fused_cache_directory(),
                os.path.join("/models", "diffusion_models", "fused_cache"),
            )

    def test_rejects_rotated_q8_cr_source_weights(self):
        with mock.patch.object(
            self.node_module,
            "gguf_sd_loader",
            return_value=({"blocks.0.weight.comfy_quant": torch.tensor(0)}, {}),
        ):
            with self.assertRaisesRegex(ValueError, "Q8_CR GGUF"):
                self.node_module._load_fusion_source("base.gguf")

    def test_materializes_standard_gguf_source(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "base.gguf"
            writer = gguf.GGUFWriter(path=None, arch="minimax_h3")
            writer.add_tensor(
                "blocks.0.attn.qkv_proj.weight",
                weight.numpy(),
                raw_dtype=gguf.GGMLQuantizationType.F16,
            )
            writer.write_header_to_file(path=str(path))
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            loaded, _ = self.node_module._load_fusion_source(str(path))
            loaded_weight = loaded.pop("blocks.0.attn.qkv_proj.weight")
            self.assertEqual(loaded_weight.dtype, torch.float16)
            self.assertTrue(torch.equal(loaded_weight, weight))
            del loaded_weight
            del loaded
            gc.collect()

    def test_imports_and_fuses_safetensors_factor_pair(self):
        down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        up = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.safetensors"
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": down,
                    "blocks.0.attn.qkv_proj.lora_B.weight": up,
                    "blocks.0.attn.qkv_proj.alpha": torch.tensor(4.0),
                },
                str(path),
            )
            _, targets, _ = load_lora(path)

        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.zeros((3, 2), dtype=torch.float32)
        }
        fuse_targets_into_state_dict(state_dict, targets, strength=0.5, device=torch.device("cpu"))
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [4.0, 6.0]])
        self.assertTrue(torch.equal(state_dict["blocks.0.attn.qkv_proj.weight"], expected))

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch does not support FP8")
    def test_fuses_scaled_fp8_target_as_fp16(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.ones(
                (2, 2), dtype=torch.float8_e4m3fn
            ),
            "blocks.0.attn.qkv_proj.weight_scale": torch.tensor(0.5),
        }
        targets = {
            "blocks.0.attn.qkv_proj": {
                "base_name": "blocks.0.attn.qkv_proj.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            }
        }

        fuse_targets_into_state_dict(state_dict, targets, strength=1.0, device=torch.device("cpu"))

        self.assertEqual(state_dict["blocks.0.attn.qkv_proj.weight"].dtype, torch.float16)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.attn.qkv_proj.weight"],
                torch.tensor([[1.5, 0.5], [0.5, 1.5]], dtype=torch.float16),
            )
        )

    def test_converter_merges_safetensors_lora_before_gguf_export(self):
        source = {
            "video_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "audio_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "blocks.0.attn.qkv_proj.weight": torch.zeros((96, 32), dtype=torch.float16),
            "final_layer.video_out.weight": torch.zeros((96, 32), dtype=torch.float16),
        }
        down = torch.zeros((2, 32), dtype=torch.float16)
        down[:, 0] = 1
        up = torch.zeros((96, 2), dtype=torch.float16)
        up[0] = 1

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            lora_path = Path(temp_dir) / "adapter.safetensors"
            output_path = Path(temp_dir) / "merged.gguf"
            save_file(source, str(source_path))
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": down,
                    "blocks.0.attn.qkv_proj.lora_B.weight": up,
                    "blocks.0.attn.qkv_proj.alpha": torch.tensor(2.0),
                },
                str(lora_path),
            )
            convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="F16",
                lora_paths=[str(lora_path)],
            )
            reader = gguf.GGUFReader(str(output_path))
            tensor = next(
                tensor for tensor in reader.tensors
                if tensor.name == "blocks.0.attn.qkv_proj.weight"
            )
            merged = torch.from_numpy(tensor.data.copy()).view(torch.float16).reshape(
                tuple(reversed(tensor.shape))
            )
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()

        self.assertTrue(torch.equal(merged[0, :2], torch.tensor([2.0, 0.0], dtype=torch.float16)))


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


class Gemma4GGUFLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_maps_e4b_specific_tensor_layout_to_comfyui(self):
        state_dict = {
            "token_embd.weight": torch.zeros(1),
            "per_layer_token_embd.weight": torch.zeros(1),
            "per_layer_model_proj.weight": torch.zeros(1),
            "per_layer_proj_norm.weight": torch.zeros(1),
            "blk.0.inp_gate.weight": torch.zeros(1),
            "blk.0.proj.weight": torch.zeros(1),
            "blk.0.layer_output_scale.weight": torch.zeros(1),
            "blk.0.post_norm.weight": torch.zeros(1),
            "blk.0.post_ffw_norm.weight": torch.zeros(1),
        }

        mapped = self.loader.sd_map_replace(state_dict, self.loader.GEMMA4_SD_MAP)

        self.assertEqual(
            set(mapped),
            {
                "model.embed_tokens.weight",
                "model.embed_tokens_per_layer.weight",
                "model.per_layer_model_projection.weight",
                "model.per_layer_projection_norm.weight",
                "model.layers.0.per_layer_input_gate.weight",
                "model.layers.0.per_layer_projection.weight",
                "model.layers.0.layer_scalar",
                "model.layers.0.post_per_layer_input_norm.weight",
                "model.layers.0.post_feedforward_layernorm.weight",
            },
        )

    def test_recreates_gemma4_bpe_tokenizer_json(self):
        tokenizer_json = self.loader.gemma4_tokenizer_json(
            ["<pad>", "<eos>", "<bos>", "<unk>", "hello", "\u2581world"],
            ["h e", "he llo"],
            [3, 3, 3, 3, 1, 1],
        )
        tokenizer = json.loads(bytes(tokenizer_json.tolist()))

        self.assertEqual(tokenizer["model"]["type"], "BPE")
        self.assertEqual(tokenizer["model"]["vocab"]["hello"], 4)
        self.assertEqual(tokenizer["pre_tokenizer"]["type"], "Metaspace")
        self.assertEqual(
            [token["content"] for token in tokenizer["added_tokens"]],
            ["<pad>", "<eos>", "<bos>", "<unk>"],
        )

    def test_e4b_layout_is_detected_by_installed_comfyui(self):
        state_dict = {
            "model.layers.0.post_feedforward_layernorm.weight": torch.zeros(2560),
            "model.layers.41.self_attn.q_norm.weight": torch.zeros(256),
        }

        self.assertEqual(
            comfy.sd.detect_te_model(state_dict),
            comfy.sd.TEModel.GEMMA_4_E4B,
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
