# SPDX-License-Identifier: Apache-2.0
"""Print a pinned launch command; --execute starts it after checking resources."""

import argparse
import json
import shlex
import socket
import subprocess
from pathlib import Path

from build_runtime import sha256


def command(runtime, model, cache, image, name, port):
    manifest = json.loads((runtime / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        if sha256(runtime / relative) != expected:
            raise ValueError(f"Runtime was modified: {relative}")
    return [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--restart=no",
        "--network=host",
        "--ipc=host",
        "--device",
        "/dev/dri:rwm",
        "--device",
        "/dev/mxcd:rwm",
        "--label",
        "org.flagos.dsv4.source=" + manifest["fingerprint"],
        "--log-opt",
        "max-size=100m",
        "--log-opt",
        "max-file=3",
        "-v",
        f"{runtime}:/dsv4:ro",
        "-v",
        f"{model}:/models/DeepSeek-V4-Flash-W8A8-INT8:ro",
        "-v",
        f"{cache}:/var/lib/dsv4:rw",
        "-e",
        "PYTHONPATH=",
        "--entrypoint",
        "/opt/conda/bin/python",
        image,
        "-S",
        "/dsv4/entrypoint.py",
        "--port",
        str(port),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", default="dsv4-w8a8-profile")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    runtime, model, cache = (
        p.resolve() for p in (args.runtime, args.model, args.cache)
    )
    if not runtime.is_dir() or not model.is_dir():
        parser.error("Runtime and model directories must exist")
    if (
        cache == runtime
        or cache.is_relative_to(runtime)
        or runtime.is_relative_to(cache)
    ):
        parser.error("Cache must be separate from runtime source")
    if cache == model or cache.is_relative_to(model) or model.is_relative_to(cache):
        parser.error("Cache must be separate from weights")
    cmd = command(runtime, model, cache, args.image, args.name, args.port)
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        return
    manifest = json.loads((runtime / "manifest.json").read_text())
    actual = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    if actual != manifest["image"]:
        raise RuntimeError(
            "Different dependency image; validate it separately before changing the pin"
        )
    if "no process found" not in subprocess.check_output(["mx-smi"], text=True):
        raise RuntimeError("GPU is busy; no workload was stopped")
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", args.port))
    if cache.exists() and any(cache.iterdir()):
        raise RuntimeError("Choose a new empty cache directory")
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
