# SPDX-License-Identifier: Apache-2.0
"""Assemble the evaluated profile from verified sources, without importing ML libraries."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath

PROFILE = Path(__file__).resolve().parent / "profile"
PACKAGES = {"vllm_fl", "flag_gems", "flaggems_vllm", "dsv4_metax"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe manifest path: {name}")
    if path.parts[0] not in {"src", "config", "entrypoint.py"}:
        raise ValueError(f"Unexpected manifest path: {name}")
    return path


def within(root, relative):
    result = root.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Source escapes its root: {relative}")
    return result


def assemble(profile, sources, output, *, thinking=False):
    profile, output = Path(profile).resolve(), Path(output).resolve()
    sources = {k: Path(v).resolve() for k, v in sources.items()}
    if set(sources) - PACKAGES:
        raise ValueError("Unknown dependency package")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for root in [profile, *sources.values()]:
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output must be separate from source inputs")
    manifest_name = "thinking-manifest.json" if thinking else "manifest.json"
    manifest = json.loads((profile / manifest_name).read_text())
    files = manifest["files"]
    expected = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if expected != manifest["fingerprint"]:
        raise ValueError("Invalid manifest fingerprint")
    resolved = {}
    for name, expected_hash in files.items():
        rel = checked_relative(name)
        source = within(profile, rel)
        if thinking and name == "config/production.json":
            source = profile / "thinking.json"
        if not source.is_file():
            if len(rel.parts) < 3 or rel.parts[0] != "src":
                raise FileNotFoundError(f"Missing profile source: {name}")
            package = rel.parts[1]
            if package not in sources:
                raise ValueError(f"Provide --source {package}=/path/to/{package}")
            source = within(sources[package], PurePosixPath(*rel.parts[2:]))
        if not source.is_file() or sha256(source) != expected_hash:
            raise ValueError(f"Dependency mismatch: {name}; use the pinned source")
        resolved[name] = source
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dsv4-build-", dir=output.parent))
    try:
        for name, source in resolved.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256(destination) != files[name]:
                raise ValueError(f"Source changed during copy: {name}")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest["fingerprint"], len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="PACKAGE=DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    sources = {}
    for value in args.source:
        key, sep, path = value.partition("=")
        if not sep or not path or key in sources:
            parser.error("Use each --source PACKAGE=DIR once")
        sources[key] = path
    fingerprint, count = assemble(PROFILE, sources, args.output, thinking=args.thinking)
    print(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "verified_files": count,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
