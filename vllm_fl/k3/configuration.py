"""Immutable K3 execution policy, derived from model and parallel configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class K3ExecutionConfig:
    enabled: bool = False
    tp8_compute: bool = False
    host_pp_feedback: bool = False
    ep_allreduce: bool = False
    feature_striped: bool = False

    @classmethod
    def from_vllm(cls, config, *, vendor):
        model = config.model_config
        hf = getattr(model, "hf_config", None)
        model_type = getattr(hf, "model_type", None)
        # vLLM constructs K3's language submodel with its own kimi_linear HF
        # config. The selected quantization backend carries the family marker.
        family = getattr(getattr(config, "quant_config", None), "model_family", None)
        is_k3 = model_type == "kimi_k3" or (
            model_type == "kimi_linear" and family == "kimi_k3"
        )
        if vendor != "metax" or not is_k3:
            return cls()
        p = config.parallel_config
        options = config.additional_config.get("k3", {})
        unknown = set(options) - {"expert_transport"}
        if unknown:
            raise ValueError(f"Unknown K3 execution options: {sorted(unknown)}")
        transport = options.get("expert_transport", "native")
        if transport not in ("native", "tp8_dp8_allreduce", "feature_striped"):
            raise ValueError(f"Unknown K3 expert transport: {transport}")
        ep_allreduce = transport != "native"
        if ep_allreduce and not (
            p.tensor_parallel_size == 8
            and p.data_parallel_size == 8
            and p.pipeline_parallel_size == 1
            and p.enable_expert_parallel
        ):
            raise ValueError("K3 explicit EP transport requires TP8/DP8/PP1/EP64")
        return cls(
            enabled=True,
            tp8_compute=p.tensor_parallel_size == 8,
            host_pp_feedback=p.pipeline_parallel_size > 1,
            ep_allreduce=ep_allreduce,
            feature_striped=transport == "feature_striped",
        )
