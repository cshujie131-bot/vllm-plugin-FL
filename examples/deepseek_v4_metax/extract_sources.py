# SPDX-License-Identifier: Apache-2.0
"""Extract unchanged dependencies from the pinned image without starting it."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from build_runtime import PROFILE, sha256

LOCATIONS = {
    "vllm_fl": [
        "/opt/vllm-plugin-FL/vllm_fl",
        "/opt/conda/lib/python3.12/site-packages/vllm_fl",
    ],
    "flag_gems": ["/opt/conda/lib/python3.12/site-packages/flag_gems"],
    "dsv4_metax": ["/independent_runtime/dsv4_metax"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    manifest = json.loads((PROFILE / "manifest.json").read_text())
    actual = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    if actual != manifest["image"]:
        raise RuntimeError("The dependency image does not match the evaluated version")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dsv4-sources-", dir=output.parent))
    container = None
    try:
        container = subprocess.check_output(
            ["docker", "create", args.image], text=True
        ).strip()
        for package, locations in LOCATIONS.items():
            required = {}
            for name, digest in manifest["files"].items():
                rel = PurePosixPath(name)
                if rel.parts[:2] == ("src", package) and not (PROFILE / name).exists():
                    required[Path(*rel.parts[2:])] = digest
            for index, location in enumerate(locations):
                candidate = staging / f"{package}-{index}"
                candidate.mkdir()
                copied = subprocess.run(
                    ["docker", "cp", "-L", f"{container}:{location}/.", str(candidate)],
                    capture_output=True,
                )
                if copied.returncode == 0 and all(
                    (candidate / p).is_file() and sha256(candidate / p) == h
                    for p, h in required.items()
                ):
                    candidate.rename(staging / package)
                    break
                shutil.rmtree(candidate)
            else:
                raise RuntimeError(f"Cannot find the pinned {package} source")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    finally:
        if container is not None:
            subprocess.run(
                ["docker", "rm", container], check=True, stdout=subprocess.DEVNULL
            )
    print(
        json.dumps({"output": str(output), "image": actual, "container_started": False})
    )


if __name__ == "__main__":
    main()
