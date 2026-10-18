# SPDX-License-Identifier: Apache-2.0
"""CPU-only publication contracts; no torch, vLLM, GPU or network imports."""

import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/deepseek_v4_metax"
PROFILE = EXAMPLE / "profile"
spec = importlib.util.spec_from_file_location(
    "dsv4_builder", EXAMPLE / "build_runtime.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fingerprint(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


class PublishedProfile(unittest.TestCase):
    def test_manifest_fingerprints(self):
        for name in ("manifest.json", "thinking-manifest.json"):
            m = json.loads((PROFILE / name).read_text())
            self.assertEqual(fingerprint(m["files"]), m["fingerprint"])

    def test_published_sources_are_unmodified(self):
        m = json.loads((PROFILE / "manifest.json").read_text())
        inventory = json.loads((PROFILE / "sources.json").read_text())
        for name in inventory["included_files"]:
            self.assertEqual(builder.sha256(PROFILE / name), m["files"][name], name)

    def test_only_service_profile_differs_for_thinking(self):
        a = json.loads((PROFILE / "manifest.json").read_text())["files"]
        b = json.loads((PROFILE / "thinking-manifest.json").read_text())["files"]
        self.assertEqual([p for p in a if a[p] != b[p]], ["config/production.json"])
        x = json.loads((PROFILE / "config/production.json").read_text())
        y = json.loads((PROFILE / "thinking.json").read_text())
        for key in x.keys() - {"service", "release"}:
            self.assertEqual(x[key], y[key], key)

    def test_effective_blacklist_and_graph_policy(self):
        p = json.loads((PROFILE / "config/production.json").read_text())
        self.assertEqual(
            p["dispatch"]["flagos_blacklist"], ["topk", "masked_fill", "masked_fill_"]
        )
        self.assertEqual(
            p["service"]["compilation_config"]["cudagraph_mode"], "FULL_DECODE_ONLY"
        )
        self.assertEqual(p["framework"]["indexer_query_shards"], 4)

    def test_fgv_mhc_provider_and_guarded_fallback(self):
        root = PROFILE / "src/flaggems_vllm/ops/mhc"
        self.assertIn(
            "from flaggems_vllm.ops.mhc.mhc_pre_metax import mhc_pre",
            (root / "__init__.py").read_text(),
        )
        adapter = (root / "mhc_pre_metax.py").read_text()
        self.assertIn("from .metax_pr755 import run", adapter)
        self.assertIn("if not eligible:", adapter)
        self.assertIn("return _legacy_mhc_pre(", adapter)

    def test_local_attention_has_no_framework_communication_imports(self):
        for name in ("flashmla_sparse.py", "flash_mla_with_kvcache.py"):
            tree = ast.parse((PROFILE / "src/flaggems_vllm/ops" / name).read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom):
                    self.assertFalse(
                        (n.module or "").startswith(("vllm_fl", "torch.distributed"))
                    )
                if isinstance(n, ast.Import):
                    for alias in n.names:
                        self.assertFalse(
                            alias.name.startswith(("vllm_fl", "torch.distributed"))
                        )

    def test_no_vendor_framework_import(self):
        for path in (PROFILE / "src").rglob("*.py"):
            for n in ast.walk(ast.parse(path.read_text())):
                if isinstance(n, ast.ImportFrom):
                    self.assertFalse(
                        (n.module or "").startswith("vllm_metax"), str(path)
                    )
                if isinstance(n, ast.Import):
                    self.assertFalse(
                        any(x.name.startswith("vllm_metax") for x in n.names), str(path)
                    )

    def test_no_private_host_paths(self):
        for path in PROFILE.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text()
            for marker in ("/data/sj/", "/Users/", "jumpserver_", "bastion.aiops"):
                self.assertNotIn(marker, text, str(path))

    def test_old_environment_bridge_removed(self):
        self.assertFalse((ROOT / "vllm_fl/ops/deepseek_v4_metax").exists())
        text = (PROFILE / "src/vllm_fl/dsv4_config.py").read_text()
        self.assertNotIn("os.environ", text)
        self.assertNotIn("os.getenv", text)

    def test_accuracy_metrics_keep_native_and_review_separate(self):
        data = json.loads((PROFILE / "validation.json").read_text())["thinking"]
        self.assertEqual(data["gpqa_strict"], [170, 198])
        self.assertEqual(data["math500_native"], [477, 500])
        self.assertEqual(data["math500_posthoc_equivalence"], [498, 500])
        self.assertIn("not the native score", data["math500_posthoc_note"])

    def test_exported_python_syntax(self):
        for path in (PROFILE / "src").rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


class BuilderSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profile = self.root / "profile"
        self.base = self.root / "base"
        self.profile.mkdir()
        self.base.mkdir()
        (self.base / "plain.py").write_text("VALUE = 1\n")
        (self.profile / "entrypoint.py").write_text("pass\n")
        self.files = {
            "src/flag_gems/plain.py": builder.sha256(self.base / "plain.py"),
            "entrypoint.py": builder.sha256(self.profile / "entrypoint.py"),
        }
        self.save_manifest()
        self.output = self.root / "assembled"

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self):
        (self.profile / "manifest.json").write_text(
            json.dumps(
                {
                    "files": self.files,
                    "fingerprint": fingerprint(self.files),
                    "image": "test",
                }
            )
        )

    def build(self, sources=None):
        return builder.assemble(
            self.profile, sources or {"flag_gems": self.base}, self.output
        )

    def test_valid_build_preserves_bytes(self):
        _, count = self.build()
        self.assertEqual(count, 2)
        self.assertEqual(
            (self.output / "src/flag_gems/plain.py").read_bytes(),
            (self.base / "plain.py").read_bytes(),
        )

    def test_mismatched_dependency_rejected_before_writes(self):
        (self.base / "plain.py").write_text("WRONG = 2\n")
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_existing_output_not_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.build()

    def test_unknown_package_rejected(self):
        with self.assertRaises(ValueError):
            self.build({"unexpected": self.base})

    def test_missing_package_rejected(self):
        with self.assertRaises(ValueError):
            builder.assemble(self.profile, {}, self.output)

    def test_path_traversal_rejected(self):
        self.files["src/../../escape"] = "0" * 64
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.build()

    def test_symlink_escape_rejected(self):
        (self.base / "plain.py").unlink()
        (self.base / "plain.py").symlink_to(self.profile / "entrypoint.py")
        self.files["src/flag_gems/plain.py"] = builder.sha256(
            self.profile / "entrypoint.py"
        )
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.build()

    def test_output_cannot_overlap_sources(self):
        self.output = self.base / "nested"
        with self.assertRaises(ValueError):
            self.build()


if __name__ == "__main__":
    unittest.main()
