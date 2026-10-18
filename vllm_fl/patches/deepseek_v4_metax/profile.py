"""One declarative release profile; no optimization environment-variable bridge."""

import json
from pathlib import Path


def load():
    import os

    path = Path(os.environ["VLLM_FL_CONFIG"])
    config = json.loads(path.read_text())
    data = config["deepseek_v4_metax"]
    if data["schema_version"] != 1:
        raise ValueError("Unsupported DS configuration schema")
    service = data["service"]
    if (
        service["tensor_parallel_size"] != 8
        or service["max_num_batched_tokens"] != 4096
    ):
        raise ValueError("This configuration is validated for TP8 / MBT4096")
    if service["compilation_config"]["cudagraph_mode"] != "FULL_DECODE_ONLY":
        raise ValueError("Prefill graph replay is not supported")
    data["dispatch"] = {
        key: value for key, value in config.items() if key != "deepseek_v4_metax"
    }
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
