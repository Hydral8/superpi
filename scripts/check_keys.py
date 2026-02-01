
from safetensors import safe_open
from huggingface_hub import hf_hub_download

repo_id = "exla-ai/openpie-0.6"
path = hf_hub_download(repo_id=repo_id, filename="policy.safetensors")

with safe_open(path, framework="pt", device="cpu") as f:
    keys = f.keys()
    print("First 20 keys:")
    for k in list(keys)[:20]:
        print(k)

    print("\n--- BASE VLM KEYS (paligemma_with_expert.paligemma) ---")
    base_keys = [k for k in keys if "paligemma_with_expert.paligemma" in k]
    print(f"Found {len(base_keys)} Base VLM keys")
    for k in base_keys[:20]:
        print(k)
