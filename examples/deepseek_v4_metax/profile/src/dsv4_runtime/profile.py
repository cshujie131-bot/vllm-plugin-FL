"""One declarative release profile; no optimization environment-variable bridge."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "config" / "production.json"


def load():
    data = json.loads(PROFILE_PATH.read_text())
    assert data["schema_version"] == 1
    service = data["service"]
    assert service["tensor_parallel_size"] == 8
    assert service["max_num_batched_tokens"] == 4096
    assert service["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert data["framework"]["indexer_query_shards"] == 4
    return data


def service_args(profile, port=None):
    s = profile["service"]
    args = ["vllm", "serve", s["model"]]
    for name, value in s.items():
        if name == "model":
            continue
        if name == "port" and port is not None:
            value = port
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(option)
        else:
            args += [
                option,
                json.dumps(value) if isinstance(value, (dict, list)) else str(value),
            ]
    return args
