
import re
import os
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

repo_id = "exla-ai/openpie-0.6"
print(f"Downloading {repo_id}...")
policy_path = hf_hub_download(repo_id=repo_id, filename="policy.safetensors")

print(f"Loading {policy_path}...")
sd = load_file(policy_path)

print("\n--- Searching for embedding-related keys ---")
# Searching for keys containing embed_tokens, token_emb, word_emb, or embedding
pattern = re.compile(r"(embed_tokens|token_emb|word_emb|embedding)", re.IGNORECASE)
cands = [k for k in sd.keys() if pattern.search(k)]

print(f"Found {len(cands)} candidates:\n")
if cands:
    print("\n".join(cands[:200]))
else:
    print("No matching keys found.")

print("\n--- Also checking for lm_head to confirm presence ---")
lm_heads = [k for k in sd.keys() if "lm_head" in k]
print(f"Found {len(lm_heads)} lm_head keys:")
for k in lm_heads:
    print(k)
