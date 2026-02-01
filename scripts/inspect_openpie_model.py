
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import torch
import sys

def inspect_file(repo_id, filename, label):
    try:
        print(f"\n--- Inspecting {label} ({filename}) ---")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        weights = load_file(path)
        num_tensors = len(weights)
        total_params = sum(t.numel() for t in weights.values())
        print(f"Num Tensors: {num_tensors}")
        print(f"Total Params: {total_params / 1e9:.2f} B")
        
        print(f"First 20 keys:")
        keys = list(weights.keys())
        for key in keys[:20]:
            print(f"  {key}")
            
        return keys
    except Exception as e:
        print(f"Error inspecting {filename}: {e}")
        return []

try:
    repo_id = "exla-ai/openpie-0.6"
    
    # Config
    print("Downloading config...")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("\nModel Config:")
    print(json.dumps(config, indent=2))

    # Policy
    policy_keys = inspect_file(repo_id, "policy.safetensors", "Policy")
    
    # Value Fn
    value_keys = inspect_file(repo_id, "value_fn.safetensors", "Value Function")

    # Heuristics
    print("\n--- Analysis ---")
    all_keys = policy_keys + value_keys
    
    has_paligemma = any("paligemma" in k.lower() or "llm" in k.lower() for k in policy_keys)
    has_gemma3 = any("gemma3" in k.lower() for k in policy_keys)
    # Pi0.5 typically uses an embedding for time in the flow matching process if explicitly named, 
    # but often it's just 'time_emb' or similar in the action expert.
    has_time_emb = any("time_emb" in k.lower() for k in policy_keys)
    
    print(f"Has PaliGemma/LLM keys: {has_paligemma}")
    print(f"Has Gemma3 keys: {has_gemma3}")
    print(f"Has Time Embeddings: {has_time_emb}")

except Exception as e:
    print(f"\nCritical Error: {e}")
    sys.exit(1)
