# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for layer ownership and local launch metadata bindings."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm_fl/ops/deepseek_v4_metax"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LayerBoundaryTests(unittest.TestCase):
    def setUp(self):
        proxy_module = load("ds_meta_proxy", ROOT / "vllm_fl/patches/triton_kernel.py")

        class JIT:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: (grid, args, kwargs)

        self.jit_type = JIT
        fake = types.ModuleType("triton")
        fake.runtime = types.SimpleNamespace(JITFunction=JIT)
        fake.autotune = lambda **options: lambda fn: types.SimpleNamespace(
            fn=fn, options=options
        )
        self.modules = patch.dict(
            sys.modules,
            {
                "triton": fake,
                "vllm_fl.patches.triton_kernel": proxy_module,
            },
        )
        self.modules.start()
        self.launch = load("ds_local_launch", OPS / "launch.py")

    def tearDown(self):
        self.modules.stop()

    def test_proxy_is_local_and_overrides_metadata(self):
        jit = self.jit_type()
        original = types.SimpleNamespace(fn=types.SimpleNamespace(fn=jit))
        bound = self.launch.local_launch(original, {"BK": 16, "num_warps": 4})
        self.assertIs(original.fn.fn, jit)
        self.assertIsNot(bound, original)
        _, args, kwargs = bound[(2,)](123, BK=64, num_warps=8)
        self.assertEqual(args, (123,))
        self.assertEqual(kwargs, {"BK": 16, "num_warps": 4})

    def test_autotuner_is_not_mutated(self):
        jit = self.jit_type()
        original = types.SimpleNamespace(fn=jit, configs=["original"])
        bound = self.launch.local_autotune(original, configs=["accepted"], key=["M"])
        self.assertEqual(original.configs, ["original"])
        self.assertIs(bound.fn, jit)
        self.assertEqual(bound.options["configs"], ["accepted"])

    def test_bad_wrapper_is_rejected(self):
        bad = types.SimpleNamespace()
        bad.fn = bad
        with self.assertRaises(TypeError):
            self.launch.local_launch(bad, {"num_warps": 4})
        with self.assertRaises(TypeError):
            self.launch.local_launch(object(), {"num_warps": 4})

    def test_examples_contain_no_source_patch_installer(self):
        example = ROOT / "examples/deepseek_v4_metax"
        self.assertFalse(list(example.rglob("*.patch")))
        self.assertFalse((example / "prepare_dependencies.py").exists())

    def test_tp_kernels_are_owned_by_fl(self):
        expected = {
            "decode.py": "_partial_model1_kernel_halfd_qkfold_blockdiag",
            "prefill.py": "triton_flash_mla_sparse_fwd_halfd_qkfold_blockdiag",
            "staged.py": "pv_replay",
        }
        for file, name in expected.items():
            tree = ast.parse((OPS / file).read_text())
            self.assertIn(
                name, {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            )
        headpack = (ROOT / "vllm_fl/ops/dsv4_headpack.py").read_text()
        self.assertIn("vllm_fl.ops.deepseek_v4_metax import prefill", headpack)
        self.assertNotIn("flaggems_vllm.ops.flashmla_sparse", headpack)

    def test_tuning_has_no_mhc_or_compiler_policy(self):
        tuning = load("ds_attention_tuning_test", OPS / "tuning.py")
        from dataclasses import asdict

        values = asdict(tuning.KernelPolicy())
        self.assertTrue(all(k.startswith(("decode_", "prefill_")) for k in values))
        self.assertEqual(values["decode_splits"], 4)

    def test_model_layer_registration_is_deferred(self):
        patches = ROOT / "vllm_fl/dispatch/backends/vendor/metax/patches/__init__.py"
        tree = ast.parse(patches.read_text())

        def imports_gdn(node):
            return isinstance(node, ast.ImportFrom) and any(
                alias.name == "gdn_linear_attn" for alias in node.names
            )

        self.assertFalse(any(imports_gdn(node) for node in tree.body))
        hook = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "register_model_patches"
        )
        self.assertTrue(any(imports_gdn(node) for node in ast.walk(hook)))
        entry = ast.parse((ROOT / "vllm_fl/__init__.py").read_text())
        model_hook = next(
            node
            for node in entry.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_model"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "register_model_patches"
                for node in ast.walk(model_hook)
            )
        )

    def test_all_python_sources_parse(self):
        for path in OPS.glob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
