# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Apply operator patches to clean copies of pinned public source checkouts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    ).strip()


def source_imports(spec, source):
    """Read two pinned upstream kernel files before patching the FGV copy.

    This is dependency preparation, not a runtime loader. The installed vLLM
    source is never modified. Exact file hashes also support official archives.
    """
    imported = spec.get("source_import")
    if imported is None:
        return []
    if source is None:
        raise ValueError("The pinned vLLM source is required for the MSA kernels")
    source = source.resolve()
    if (source / ".git").exists() and git(source, "rev-parse", "HEAD") != imported[
        "commit"
    ]:
        raise ValueError("vLLM source commit does not match the dependency manifest")
    result = []
    for row in imported["files"]:
        src, dst = Path(row["source"]), Path(row["destination"])
        if (
            src.is_absolute()
            or dst.is_absolute()
            or ".." in src.parts
            or ".." in dst.parts
        ):
            raise ValueError("Source import paths must stay inside their checkouts")
        content = (source / src).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Upstream kernel source differs: " + str(src))
        result.append((dst, content))
    return result


def prepare(source, output, spec, *, import_source=None):
    source = source.resolve()
    if git(source, "rev-parse", "HEAD") != spec["commit"]:
        raise ValueError("Source commit does not match the dependency manifest")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("The supplied source has tracked modifications")
    imported_files = source_imports(spec, import_source)
    patches = [(HERE / "dependencies" / name).resolve() for name in spec["patches"]]
    expected = {
        "commit": spec["commit"],
        "source_import": spec.get("source_import"),
        "patches": {
            str(p.relative_to(HERE / "dependencies")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in patches
        },
    }
    marker = output / ".m3-prepared.json"
    if output.exists():
        raise FileExistsError("Choose a new output directory: " + str(output))
    subprocess.run(
        ["git", "clone", "--no-hardlinks", str(source), str(output)], check=True
    )
    subprocess.run(
        ["git", "-C", str(output), "checkout", "--detach", spec["commit"]], check=True
    )
    for relative, content in imported_files:
        target = output / relative
        if target.exists():
            raise FileExistsError(
                "Source import would overwrite a library file: " + str(relative)
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for patch in patches:
        subprocess.run(
            ["git", "-C", str(output), "apply", "--check", str(patch)], check=True
        )
        subprocess.run(["git", "-C", str(output), "apply", str(patch)], check=True)
    subprocess.run(["git", "-C", str(output), "diff", "--check"], check=True)
    marker.write_text(json.dumps(expected, indent=2) + "\n")
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flag-gems-source", type=Path, required=True)
    p.add_argument("--fgv-source", type=Path, required=True)
    p.add_argument("--vllm-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--install", action="store_true")
    a = p.parse_args(argv)
    manifest = json.loads((HERE / "dependencies/manifest.json").read_text())
    root = a.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("The dependency output directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for key, source in [
        ("flag_gems", a.flag_gems_source),
        ("flaggems_vllm", a.fgv_source),
    ]:
        prepared.append(
            prepare(source, root / key, manifest[key], import_source=a.vllm_source)
        )
    if a.install:
        for path in prepared:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "-e",
                    str(path),
                ],
                check=True,
            )
    print(
        json.dumps(
            {"prepared": [str(p) for p in prepared], "installed": a.install}, indent=2
        )
    )


if __name__ == "__main__":
    main()
