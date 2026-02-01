

import os
import sys
import logging
import types
import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Add src to path
sys.path.insert(0, "lib") # fallback
sys.path.insert(0, "src")

# Now import the model components which import JAX config
from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.models import model as _model

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    force=True,  # overrides the WARNING-level config that already exists
)

# Setup logging
logger = logging.getLogger(__name__)

def main():
    print("Starting OpenPie-0.6 Inference (PyTorch)...")
    logger.info("Starting OpenPie-0.6 Inference (PyTorch)...")
    
    # Check Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    repo_id = "exla-ai/openpie-0.6"
    
    # Download weights
    logger.info(f"Downloading model from {repo_id}...")
    try:
        policy_path = hf_hub_download(repo_id=repo_id, filename="policy.safetensors")
        logger.info(f"Policy weights at: {policy_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return

    # Define Config - Pi0.5 Architecture
    # UPDATED based on weight size mismatch errors:
    # Expert width 2048 -> gemma_2b
    # Action dim 14 -> 14
    logger.info("Initializing Pi0.5 Config (Updated to match weights)...")
    model_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_2b", # Was gemma_300m, but weights are 2048 width
        action_dim=14,                    # Was 7, but weights are 14 dim
        action_horizon=50,
        dtype="bfloat16"
    )
    
    # Initialize Model
    logger.info("Creating PI0Pytorch model (on meta device)...")
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = pi0_pytorch.PI0Pytorch(model_config)
    
    # Materialize model on device
    logger.info(f"Materializing model on {device}...")
    model.to_empty(device=device)
    
    # Load OpenPie Policy Weights
    print("LOADING POLICY WEIGHTS")
    logger.info(f"Loading weights from {policy_path}...")
    from safetensors.torch import load_model
    
    # Load policy weights with "Safe Strict" verification
    logger.info("Loading policy weights...")
    try:
        load_model(model, policy_path, strict=True)
    except Exception as e:
        if "Missing key(s)" in str(e):
            # Check if ONLY embed_tokens is missing (common with tied weights)
            logger.info("Strict load failed. verifying missing keys...")
            # We need to manually load to inspect missing keys properly
            from safetensors.torch import load_file
            state_dict = load_file(policy_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            allowed_missing = ["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"]
            real_missing = [k for k in missing if k not in allowed_missing]
            
            if real_missing:
                logger.error(f"CRITICAL: Real keys missing: {real_missing}")
                raise e
            if unexpected:
                 logger.error(f"CRITICAL: Unexpected keys found: {unexpected}")
                 raise e
                 
            logger.info("Verified: Only tied embedding weights are missing. Load successful.")
            
            # CRITICAL: Manually tie embeddings if not tied by default
            # Checkpoint only has lm_head.weight, so we copy it to embed_tokens
            vlm = model.paligemma_with_expert.paligemma
            logger.info("Tying VLM embeddings to lm_head...")
            vlm.get_input_embeddings().weight.data = vlm.get_output_embeddings().weight.data.clone()
        else:
            raise e
    model.to(device)
    model.eval()

    print("Beginning inference")
    
    # Inputs
    prompt = "pour a cup of coffee"
    logger.info(f"Prompt: '{prompt}'")
    
    # Construct Inputs
    batch_size = 1
    
    images = {
        "base_0_rgb": torch.zeros((batch_size, 3, 224, 224), device=device, dtype=torch.float32),
        "left_wrist_0_rgb": torch.zeros((batch_size, 3, 224, 224), device=device, dtype=torch.float32),
        "right_wrist_0_rgb": torch.zeros((batch_size, 3, 224, 224), device=device, dtype=torch.float32)
    }
    
    image_masks = {
        k: torch.ones((batch_size,), device=device, dtype=torch.bool) for k in images
    }
    
    # State needs to match updated action dim if used in projection?
    # In pi0_pytorch.py:
    # if pi05:
    #   time_mlp_in = Linear(expert_width, expert_width)
    # else:
    #   state_proj = Linear(32, expert_width)
    #
    # Wait, Pi05 handles state differently (as tokens). 
    # model.pi05 is True.
    # So `state` input to Observation might just be passed through but converted.
    # Let's keep state dim generic, Observation expects it.
    state = torch.zeros((batch_size, 14), device=device, dtype=torch.float32) 
    
    # Tokenize Prompt
    # Tokenize Prompt
    from openpi.models import tokenizer
    tok = tokenizer.PaligemmaTokenizer(model_config.max_token_len)
    
    # Needs state for Pi0.5 formatting!
    state_np = state.cpu().numpy()[0] # [14]
    
    tokenized_prompt_np, mask_np = tok.tokenize(prompt, state=state_np) 
    
    tokenized_prompt = torch.from_numpy(tokenized_prompt_np).to(device).unsqueeze(0) # Add batch dim
    tokenized_prompt_mask = torch.from_numpy(mask_np).to(device).unsqueeze(0) # Add batch dim
    
    # Construct Observation
    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask
    )
    
    # Run Inference
    logger.info("Running sample_actions...")
    with torch.no_grad():
        actions = model.sample_actions(device, obs, num_steps=10)
        
    logger.info("Inference Complete!")
    logger.info(f"Action Shape: {actions.shape}") 
    
    # Compute stats
    actions_cpu = actions.cpu().numpy()
    logger.info(f"Action Mean: {np.mean(actions_cpu):.4f}")
    logger.info(f"Action Std: {np.std(actions_cpu):.4f}")
    
    # Check for validity
    if np.isnan(actions_cpu).any():
        logger.error("WARNING: NaN values detected in actions!")
    else:
        logger.info("Action states appear valid (non-NaN).")

if __name__ == "__main__":
    main()
