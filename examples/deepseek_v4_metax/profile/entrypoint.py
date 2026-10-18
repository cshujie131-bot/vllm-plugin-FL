"""Verify the sealed source/config, initialize isolated caches, and start vLLM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
IMAGE_ROOT=Path("/opt/dsv4")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int)
    parser.add_argument("--probe",action="store_true")
    parser.add_argument("--script",help="run an isolated validation script instead of vLLM")
    args=parser.parse_args()
    manifest=json.loads((ROOT/"manifest.json").read_text())
    for rel,h in manifest["files"].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    profile=json.loads((ROOT/"config/production.json").read_text())
    if not args.probe and not args.script:
        requirements=json.loads((IMAGE_ROOT/"model_requirements.json").read_text())
        model=Path(profile["service"]["model"])
        for name,h in requirements["metadata_sha256"].items():
            assert hashlib.sha256((model/name).read_bytes()).hexdigest()==h,name
        for name,size in requirements["weight_files_bytes"].items():
            assert (model/name).stat().st_size==size,name
    cache=Path("/var/lib/dsv4");cache.mkdir(parents=True,exist_ok=True)
    marker=cache/"RELEASE.json"
    expected={"release":profile["release"],"fingerprint":manifest["fingerprint"]}
    if marker.exists():
        assert json.loads(marker.read_text())==expected,"Cache release mismatch"
    else:
        assert not any(cache.iterdir()),"Refusing a nonempty unrecognized cache"
        for d in ("triton","torchinductor","vllm","flaggems","flaggems_vllm","startup_audit"):
            (cache/d).mkdir()
        shutil.copytree(IMAGE_ROOT/"flaggems_seed/config_cache",cache/"flaggems/config_cache")
        marker.write_text(json.dumps(expected,indent=2))
    # Remove inherited experimental knobs; typed policy is the only source.
    for key in list(os.environ):
        if key.startswith(("VLLM_FL_METAX_","DSV4_","MACA_VLLM_ENABLE_MCTLASS_")) or key in (
            "VLLM_FL_FLAGOS_BLACKLIST","VLLM_FL_FLAGOS_WHITELIST",
            "FLAGGEMS_VLLM_ENABLE_METAX_GDN_COMPILER_PATCH"):
            os.environ.pop(key)
    os.environ.update(profile["infrastructure_env"])
    os.environ.update({
        "PYTHONPATH":str(ROOT/"src"),
        "VLLM_FL_CONFIG":str(ROOT/"config/dispatch.yaml"),
        "VLLM_TUNED_CONFIG_FOLDER":profile["moe"]["tuned_config_folder"],
        "TRITON_CACHE_DIR":str(cache/"triton"),
        "TORCHINDUCTOR_CACHE_DIR":str(cache/"torchinductor"),
        "VLLM_CACHE_ROOT":str(cache/"vllm"),
        "FLAGGEMS_CACHE_DIR":str(cache/"flaggems"),
        "FLAGGEMS_VLLM_CACHE_DIR":str(cache/"flaggems_vllm"),
    })
    if args.probe or args.script:
        cmd=["/opt/conda/bin/python",args.script or "/work/tests/candidate_probe.py"]
    else:
        sys.path.insert(0,str(ROOT/"src"))
        from dsv4_runtime.profile import service_args
        cmd=service_args(profile,args.port)
    print("DSV4_CLEAN_ENTRY "+json.dumps({"release":profile["release"],"command":cmd}),flush=True)
    os.execvpe(cmd[0],cmd,os.environ)

if __name__=="__main__":main()
