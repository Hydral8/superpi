
import os
import sys
import logging
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model

# Add src to path
sys.path.insert(0, "lib")
sys.path.insert(0, "src")

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    repo_id = "exla-ai/openpie-0.6"
    policy_path = hf_hub_download(repo_id=repo_id, filename="policy.safetensors")

    model_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_2b",
        action_dim=14,
        action_horizon=50,
        dtype="bfloat16"
    )
    
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = pi0_pytorch.PI0Pytorch(model_config)
    
    # Materialize on CPU for easier data_ptr check without risking OOM during check 
    # (actually A100 has plenty, but CPU is safer for a quick check)
    model.to_empty(device="cpu")
    
    logger.info("Loading weights with strict=False...")
    from safetensors.torch import load_file
    state_dict = load_file(policy_path)
    model.load_state_dict(state_dict, strict=False)
    
    # Validation step provided by USER
    print("\n--- VALIDATING TIED EMBEDDINGS ---")
    try:
        vlm = model.paligemma_with_expert.paligemma
        
        # Use standard HF methods to get embeddings
        emb_layer = vlm.get_input_embeddings()
        head_layer = vlm.get_output_embeddings()
        
        emb = emb_layer.weight
        head = head_layer.weight

        print(f"input_embeddings type: {type(emb_layer)}")
        print(f"output_embeddings type: {type(head_layer)}")
        print("embed shape:", tuple(emb.shape))
        print("lm_head shape:", tuple(head.shape))
        
        # Check storage
        is_tied = emb.data_ptr() == head.data_ptr()
        print("tied storage:", is_tied)
        
        # Check closeness
        is_allclose = torch.allclose(emb, head)
        print("allclose:", is_allclose)
        
        if is_tied or is_allclose:
            print("\nSUCCESS: Embeddings are correctly tied or loaded.")
        else:
            print("\nWARNING: Embeddings are NOT tied and differ! Language conditioning may be broken.")
            
    except Exception as e:
        print(f"Error accessing attributes: {e}")
        # Print actual structure to help debug
        print("\nModel structure:")
        print(f"vlm type: {type(vlm)}")

if __name__ == "__main__":
    main()
