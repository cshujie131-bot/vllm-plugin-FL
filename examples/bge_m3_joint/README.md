# BGE-M3 joint output example

See [the model guide](../../docs/models/bge_m3.md) for the output contract and validation.
Use the existing platform runtime and set:

```bash
export VLLM_PLUGINS=fl,bge_m3
export VLLM_FL_CONFIG=/path/to/vllm-plugin-FL/vllm_fl/dispatch/config/bge_m3_metax.yaml
vllm serve /models/bge-m3 --runner pooling --dtype float32 --enforce-eager \
  --hf-overrides '{"architectures":["BgeM3EmbeddingModelFL"]}' \
  --pooler-config '{"task":"embed"}' --io-processor-plugin bge_m3
```
