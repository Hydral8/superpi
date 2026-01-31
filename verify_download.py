
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import json
import os

# Download model files
print("Downloading config...")
config_path = hf_hub_download(repo_id="exla-ai/openpie-0.6", filename="config.json")
print(f"Config path: {config_path}")

with open(config_path, 'r') as f:
    config_data = json.load(f)
    print("Config data:", json.dumps(config_data, indent=2))

print("Downloading policy...")
policy_path = hf_hub_download(repo_id="exla-ai/openpie-0.6", filename="policy.safetensors")
# value_path = hf_hub_download(repo_id="exla-ai/openpie-0.6", filename="value_fn.safetensors")

# Load weights
print("Loading policy weights...")
policy_weights = load_file(policy_path)
# value_weights = load_file(value_path)

print(f"Policy model: {len(policy_weights)} tensors")
# Print some keys to check structure
print("First 10 keys:")
for k in list(policy_weights.keys())[:10]:
    print(k)

