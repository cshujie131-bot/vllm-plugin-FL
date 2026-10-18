# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned operator checkouts; do not import Torch or modify installed sources."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "dependencies" / "lock.json"


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".json", ".yaml", ".yml"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Source symlink is not supported: {path}")
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def prepare(sources, output, lock_path=LOCK):
    lock_path = Path(lock_path).resolve()
    lock = json.loads(lock_path.read_text())
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if set(sources) != set(lock):
        raise ValueError(f"Provide sources for {sorted(lock)}")
    resolved = {name: Path(path).resolve() for name, path in sources.items()}
    for name, root in resolved.items():
        if root == output or root in output.parents or output in root.parents:
            raise ValueError("Output must be separate from the source checkouts")
        if source_hash(root / "src" / name) != lock[name]["source_sha256"]:
            raise ValueError(f"{name}: source does not match {lock[name]['commit']}")
        if not (root / "pyproject.toml").is_file():
            raise ValueError(f"Not a source checkout: {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dsv4-deps-", dir=output.parent))
    try:
        for name, root in resolved.items():
            target = staging / name
            shutil.copytree(
                root,
                target,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", "build", "dist"
                ),
            )
            for filename in lock[name]["patches"]:
                patch = lock_path.parent / filename
                subprocess.run(
                    ["git", "apply", "--check", str(patch)], cwd=target, check=True
                )
                subprocess.run(["git", "apply", str(patch)], cwd=target, check=True)
            if source_hash(target / "src" / name) != lock[name]["result_sha256"]:
                raise ValueError(f"{name}: patched source verification failed")
        (staging / "prepared.json").write_text(json.dumps(lock, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flag-gems-source", type=Path, required=True)
    parser.add_argument("--fgv-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Editable-install the prepared sources in this Python environment",
    )
    args = parser.parse_args()
    output = prepare(
        {"flag_gems": args.flag_gems_source, "flaggems_vllm": args.fgv_source},
        args.output,
    )
    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(output / name),
        ]
        for name in ("flag_gems", "flaggems_vllm")
    ]
    for command in commands:
        print(" ".join(command), flush=True)
        if args.install:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
