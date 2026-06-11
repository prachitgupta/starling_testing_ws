# Fine-Tuning Quick Commands

## Hugging Face Auth

Install the Hugging Face CLI:

```bash
python -m pip install -U huggingface_hub
```

Login once and save the token on the machine:

```bash
export HF_HOME=$HOME/.cache/huggingface
huggingface-cli login
```

For DeltaAI project storage:

```bash
export HF_HOME=/projects/bhkj/$USER/hf_cache
huggingface-cli login
```

Or set a token for only the current shell:

```bash
export HF_TOKEN=hf_your_token_here
```

Check gated Llama access:

```bash
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("meta-llama/Meta-Llama-3.1-8B-Instruct").modelId)
PY
```

Do not commit Hugging Face tokens.
