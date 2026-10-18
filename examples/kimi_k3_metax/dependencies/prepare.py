"""Materialize the pinned public sources in a new, independent directory."""

import argparse
import json
import subprocess
from pathlib import Path


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error("Use a new destination; existing source trees are never reset.")
    destination.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    lock = json.loads((here / "lock.json").read_text())
    for name, dependency in lock["dependencies"].items():
        target = destination / name
        run("git", "clone", "--filter=blob:none", dependency["repository"], str(target))
        run("git", "checkout", "--detach", dependency["commit"], cwd=target)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=target, text=True
        ).strip()
        if actual != dependency["commit"]:
            raise RuntimeError(f"{name}: unexpected checkout {actual}")
        for patch in dependency["patches"]:
            run("git", "apply", "--check", str(here / patch), cwd=target)
            run("git", "apply", str(here / patch), cwd=target)
        print(f"{name}: {actual}, {len(dependency['patches'])} patches applied")


if __name__ == "__main__":
    main()
