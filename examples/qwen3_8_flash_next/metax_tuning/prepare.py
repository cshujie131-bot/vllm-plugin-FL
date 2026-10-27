#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare isolated, fingerprint-checked dependencies for the reviewed tuning."""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(sources, output, bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    # Validate every input before copying or modifying anything.
    for package, record in manifest["packages"].items():
        source = sources[package].resolve()
        if not (source / "__init__.py").is_file():
            raise ValueError(f"Expected a package directory: {source}")
        for row in record["files"]:
            path = source / row["path"]
            if not path.is_file() or digest(path) != row["before_sha256"]:
                raise ValueError(f"Source fingerprint mismatch: {path}")
    output.mkdir(parents=True)
    for package, record in manifest["packages"].items():
        destination = output / package
        shutil.copytree(
            sources[package],
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
        )
        patch = bundle / record["patch"]
        subprocess.run(
            ["git", "apply", "--check", str(patch)], cwd=destination, check=True
        )
        subprocess.run(["git", "apply", str(patch)], cwd=destination, check=True)
        for row in record["files"]:
            if digest(destination / row["path"]) != row["after_sha256"]:
                raise RuntimeError(f"Prepared fingerprint mismatch: {row['path']}")
    environment = {
        "PYTHONPATH": str(output),
        "VLLM_TUNED_CONFIG_FOLDER": str(bundle / "moe-configs"),
    }
    # FG and FGV share these override names; retain separate native caches.
    unset_environment = ["FLAGGEMS_CACHE_DIR", "FLAGGEMS_DB_URL", "GEMS_CACHE_PATH"]
    lines = ["unset " + " ".join(unset_environment)]
    for name, value in environment.items():
        suffix = "${PYTHONPATH:+:$PYTHONPATH}" if name == "PYTHONPATH" else ""
        lines.append(f"export {name}={shlex.quote(value)}{suffix}")
    (output / "env.sh").write_text("\n".join(lines) + "\n")
    (output / "prepared.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "sources": {
                    name: str(path.resolve()) for name, path in sources.items()
                },
                "environment": environment,
                "unset_environment": unset_environment,
            },
            indent=2,
        )
        + "\n"
    )
    return output / "env.sh"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flag-gems-package", type=Path, required=True)
    parser.add_argument("--fgv-package", type=Path, required=True)
    parser.add_argument("--vllm-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "flag_gems": args.flag_gems_package,
        "flaggems_vllm": args.fgv_package,
        "vllm": args.vllm_package,
    }
    print(prepare(sources, args.output.resolve(), Path(__file__).resolve().parent))


if __name__ == "__main__":
    main()
