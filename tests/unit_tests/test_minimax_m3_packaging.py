# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""CPU tests for the M3 profile, explicit blacklists and isolated launcher."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/minimax_m3_metax"


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = json.loads((EXAMPLE / "config.json").read_text())
        self.path = self.root / "profile.json"
        self.path.write_text(json.dumps(self.data))
        self.env = patch.dict(os.environ, {"VLLM_FL_CONFIG": str(self.path)})
        self.env.start()
        self.config = load_file(
            ROOT / "vllm_fl/minimax_m3_config.py", "_m3_config_test"
        )

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.data))
        self.config.load.cache_clear()

    def test_validated_defaults_and_immutable_policy(self):
        p = self.config.load()
        self.assertEqual(p.framework.tensor_parallel_size, 16)
        self.assertEqual(p.service["max_model_len"], 131072)
        self.assertEqual(p.service["max_num_batched_tokens"], 8192)
        self.assertEqual(p.service["gpu_memory_utilization"], 0.96)
        self.assertEqual(len(p.linear["legacy"]), 35)
        self.assertEqual(len(p.linear["small"]), 28)
        self.assertEqual(p.dispatch["flagos_blacklist"], ("masked_fill",))
        with self.assertRaises(TypeError):
            p.attention["base_prefill"]["warps"] = 8

    def test_only_explicit_profile_enables_specialization(self):
        with patch.dict(os.environ, {"VLLM_FL_CONFIG": ""}):
            self.assertFalse(self.config.is_enabled())
        self.data["model_profile"] = "another_model"
        self.write()
        self.assertFalse(self.config.is_enabled())
        with self.assertRaises(ValueError):
            self.config.load()

    def test_bool_is_not_integer(self):
        self.data["minimax_m3"]["framework"]["flagos_msa"] = 1
        self.write()
        with self.assertRaises(TypeError):
            self.config.load()

    def test_wrong_tp_is_rejected(self):
        self.data["minimax_m3"]["framework"]["tensor_parallel_size"] = 8
        self.write()
        with self.assertRaises(ValueError):
            self.config.load()

    def test_denied_math_path_is_rejected(self):
        self.data["minimax_m3"]["kernels"]["safe_topk"] = False
        self.write()
        with self.assertRaises(ValueError):
            self.config.load()

    def test_tle_alignment_is_not_reenabled(self):
        self.data["minimax_m3"]["kernels"]["moe_alignment_tle"] = True
        self.write()
        with self.assertRaises(ValueError):
            self.config.load()

    def test_missing_field_is_rejected(self):
        del self.data["minimax_m3"]["framework"]["max_context"]
        self.write()
        with self.assertRaises(ValueError):
            self.config.load()

    def test_duplicate_linear_shape_is_rejected(self):
        rows = self.data["minimax_m3"]["linear"]["small"]
        rows[1] = copy.deepcopy(rows[0])
        self.write()
        with self.assertRaises(ValueError):
            self.config.load()

    def test_blacklist_cannot_be_reset(self):
        self.data["flagos_blacklist"] = []
        self.write()
        with self.assertRaises(ValueError):
            self.config.load()

    def test_project_env_cannot_replace_configuration(self):
        self.data["minimax_m3"]["infrastructure_env"]["M3_MSA_SAFE_TOPK"] = "0"
        self.write()
        with self.assertRaises(ValueError):
            self.config.load()

    def dispatch_utils(self):
        return load_file(ROOT / "vllm_fl/dispatch/config/utils.py", "_m3_dispatch_test")

    def test_explicit_blacklist_beats_platform_default(self):
        mocks = {
            "yaml": SimpleNamespace(safe_load=json.loads),
            "vllm_fl.minimax_m3_config": self.config,
            "vllm_fl.utils": SimpleNamespace(VENDOR_DEVICE_MAP={"metax": "cuda"}),
        }
        with patch.dict(sys.modules, mocks):
            module = self.dispatch_utils()
            with patch.object(
                module,
                "load_platform_config",
                return_value={"flagos_blacklist": ["topk"], "oot_blacklist": ["topk"]},
            ):
                self.assertEqual(module.get_flagos_blacklist(), ["masked_fill"])
                self.assertEqual(module.get_oot_blacklist(), ["masked_fill"])

    def test_other_profiles_keep_default_blacklists(self):
        mocks = {
            "yaml": SimpleNamespace(safe_load=json.loads),
            "vllm_fl.minimax_m3_config": self.config,
            "vllm_fl.utils": SimpleNamespace(VENDOR_DEVICE_MAP={"metax": "cuda"}),
        }
        with (
            patch.dict(sys.modules, mocks),
            patch.dict(os.environ, {"VLLM_FL_CONFIG": ""}),
        ):
            module = self.dispatch_utils()
            with patch.object(
                module,
                "load_platform_config",
                return_value={"flagos_blacklist": ["topk"], "oot_blacklist": ["topk"]},
            ):
                self.assertEqual(module.get_flagos_blacklist(), ["topk"])
                self.assertEqual(module.get_oot_blacklist(), ["topk"])


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.compiler = self.root / "compiler"
        (self.compiler / "triton").mkdir(parents=True)
        (self.compiler / "triton/__init__.py").touch()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps({"architectures": ["MiniMaxM3SparseForConditionalGeneration"]})
        )
        self.module = load_file(EXAMPLE / "serve.py", "_m3_serve_test")
        self.args = [
            "--model",
            str(self.model),
            "--flagtree-root",
            str(self.compiler),
            "--cache-dir",
            str(self.root / "cache"),
            "--node-rank",
            "0",
            "--master-addr",
            "192.0.2.1",
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, args):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(self.module, "source_identity", return_value="test-source"),
        ):
            return self.module.main(args)

    def test_print_only_has_no_cache_or_service_side_effect(self):
        with patch.object(self.module.subprocess, "run") as run:
            self.run_main(self.args)
            run.assert_not_called()
        self.assertFalse((self.root / "cache").exists())

    def test_legacy_env_removed_and_caches_separate(self):
        cfg = json.loads((EXAMPLE / "config.json").read_text())
        with patch.dict(
            os.environ,
            {
                "M3_MSA_SAFE_TOPK": "0",
                "VLLM_FL_FLAGOS_BLACKLIST": "topk",
                "VLLM_FL_FLAGOS_BLACKLIST_APPEND": "masked_fill_",
            },
        ):
            env = self.module.build_environment(
                cfg,
                self.root / "profile",
                self.root / "state",
                self.compiler,
                EXAMPLE / "tuning-ar512k.cfg",
            )
        self.assertNotIn("M3_MSA_SAFE_TOPK", env)
        self.assertNotIn("VLLM_FL_FLAGOS_BLACKLIST", env)
        self.assertNotIn("VLLM_FL_FLAGOS_BLACKLIST_APPEND", env)
        self.assertNotEqual(env["FLAGGEMS_CACHE_DIR"], env["FLAGGEMS_VLLM_CACHE_DIR"])
        self.assertEqual(env["VLLM_PLUGINS"], "fl")
        self.assertEqual(env["FLAGTREE_ROOT"], str(self.compiler))
        self.assertEqual(env["PYTHONPATH"], str(self.compiler))

    def test_other_model_is_rejected_before_launch(self):
        (self.model / "config.json").write_text(
            json.dumps({"architectures": ["OtherModel"]})
        )
        with patch.object(self.module.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                self.run_main(self.args)
            run.assert_not_called()

    def test_nonempty_unknown_cache_is_preserved(self):
        state = self.root / "cache/node0"
        state.mkdir(parents=True)
        file = state / "owned-by-other-run"
        file.write_text("preserve")
        with patch.object(self.module.subprocess, "run") as run:
            with self.assertRaises(RuntimeError):
                self.run_main(self.args + ["--execute"])
            run.assert_not_called()
        self.assertEqual(file.read_text(), "preserve")

    def test_cache_identity_mismatch_is_rejected(self):
        with patch.object(self.module.subprocess, "run"):
            self.run_main(self.args + ["--execute"])
        marker = self.root / "cache/node0/CACHE_ID.json"
        data = json.loads(marker.read_text())
        data["fl_source"] = "different"
        marker.write_text(json.dumps(data))
        with patch.object(self.module.subprocess, "run") as run:
            with self.assertRaises(RuntimeError):
                self.run_main(self.args + ["--execute"])
            run.assert_not_called()


class VisionDispatchTests(unittest.TestCase):
    def test_no_private_vision_rope_implementation_or_registration(self):
        for relative in (
            "vllm_fl/ops/minimax_m3/vision.py",
            "vllm_fl/ops/minimax_m3/torch_ops.py",
            "vllm_fl/ops/minimax_m3/registration.py",
        ):
            with self.subTest(path=relative):
                source = (ROOT / relative).read_text()
                self.assertNotIn("m3_vision_rope", source)
                self.assertNotIn('"vision_rope"', source)

    def test_native_vision_attention_method_is_preserved(self):
        import ast

        source = (ROOT / "vllm_fl/models/minimax_m3/vision.py").read_text()
        classes = {
            n.name for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
        }
        self.assertNotIn("MiniMaxVLAttention", classes)
        bindings = (ROOT / "vllm_fl/models/minimax_m3/bindings.py").read_text()
        self.assertNotIn('"MiniMaxVLAttention"', bindings)
        worker = (ROOT / "vllm_fl/worker/worker.py").read_text()
        self.assertIn("patch_oot_apply_rotary_emb()", worker)


class SourceImportTests(unittest.TestCase):
    def setUp(self):
        import hashlib

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.file = self.root / "vllm/kernel.py"
        self.file.parent.mkdir()
        self.content = b"# upstream kernel fixture\n"
        self.file.write_bytes(self.content)
        self.preparer = load_file(
            EXAMPLE / "prepare_dependencies.py", "_m3_prepare_test"
        )
        self.spec = {
            "source_import": {
                "commit": "pinned",
                "files": [
                    {
                        "source": "vllm/kernel.py",
                        "destination": "src/flaggems_vllm/ops/kernel.py",
                        "sha256": hashlib.sha256(self.content).hexdigest(),
                    }
                ],
            }
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_source_import_keeps_source_read_only(self):
        result = self.preparer.source_imports(self.spec, self.root)
        self.assertEqual(
            result, [(Path("src/flaggems_vllm/ops/kernel.py"), self.content)]
        )
        self.assertEqual(self.file.read_bytes(), self.content)
        self.assertFalse((self.root / "src").exists())

    def test_changed_upstream_source_is_rejected(self):
        self.file.write_bytes(b"locally modified")
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.preparer.source_imports(self.spec, self.root)

    def test_paths_cannot_escape_the_checkout(self):
        for field in ("source", "destination"):
            with self.subTest(field=field):
                spec = copy.deepcopy(self.spec)
                spec["source_import"]["files"][0][field] = "../outside.py"
                with self.assertRaisesRegex(ValueError, "inside their checkouts"):
                    self.preparer.source_imports(spec, self.root)

    def test_missing_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vLLM source is required"):
            self.preparer.source_imports(self.spec, None)

    def test_wrong_checkout_revision_is_rejected(self):
        (self.root / ".git").mkdir()
        with patch.object(self.preparer, "git", return_value="wrong-revision"):
            with self.assertRaisesRegex(ValueError, "commit does not match"):
                self.preparer.source_imports(self.spec, self.root)

    def test_other_dependencies_do_not_require_source_imports(self):
        self.assertEqual(self.preparer.source_imports({}, None), [])


if __name__ == "__main__":
    unittest.main()
