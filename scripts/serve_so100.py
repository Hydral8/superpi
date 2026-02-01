
import logging
import os
import sys
import dataclasses
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model, load_file

# Add src to path
sys.path.insert(0, "lib")
sys.path.insert(0, "src")

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
from openpi import transforms as _transforms
from openpi.shared import normalize as _normalize

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class SO100InputTransform(_transforms.DataTransformFn):
    """Prepares observations for the SO-100 model.
    Duplicating/Blacking out views as needed.
    """
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        logger.info("Transforming observation...")
        # data usually contains 'images' (dict) and 'state' (np.array)
        # We need to map to the internal model keys: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
        
        # 1. Handle Images
        # User provides base and right wrist
        base = data.get("base_0_rgb")
        if base is None:
             # Fallback: maybe they just sent image
             base = data.get("image", np.zeros((224, 224, 3), dtype=np.uint8))
             
        right_wrist = data.get("right_wrist_0_rgb")
        if right_wrist is None:
            # Fallback to black if no wrist cam
            right_wrist = np.zeros_like(base)
            
        # Left wrist is always black for SO-100 single arm
        left_wrist = np.zeros_like(base)
        
        data["image"] = {
            "base_0_rgb": base,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        data["image_mask"] = {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": True,
        }
        
        # 2. Handle State
        # Ensure state is 14-dim (padded)
        state = data.get("state")
        if state is None:
            state = np.zeros(14)
        elif len(state) == 6:
            # Remap 6-DOF to Right Arm 7-DOF
            # SO-100: [waist, shoulder, elbow, pitch, roll, gripper]
            # ALOHA:  [waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_roll, gripper]
            state_7 = np.zeros(7)
            state_7[[0, 1, 2, 4, 5, 6]] = state
            
            new_state = np.zeros(14)
            new_state[7:] = state_7 # Right Arm
            state = new_state
        elif len(state) < 14:
            new_state = np.zeros(14)
            new_state[:len(state)] = state
            state = new_state
        data["state"] = state
        
        return data

@dataclasses.dataclass(frozen=True)
class SO100OutputRemap(_transforms.DataTransformFn):
    """Remaps 14-dim ALOHA actions to 6-DOF SO-100 actions."""
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        logger.info("Remapping actions to 6-DOF...")
        # data["actions"] is (horizon, 14)
        actions = data["actions"]
        
        # Right Arm is indices 7-13: [Waist, Shld, Elbow, ForearmRoll, Pitch, Roll, Grip]
        right_arm = actions[:, 7:14]
        
        # Remap to SO-100 6-DOF: Drop Index 3 (Forearm Roll)
        # Result: [Waist(0), Shld(1), Elbow(2), Pitch(4), Roll(5), Grip(6)]
        so100_actions = right_arm[:, [0, 1, 2, 4, 5, 6]]
        
        data["actions"] = so100_actions
        return data

def load_so100_model(device):
    repo_id = "exla-ai/openpie-0.6"
    logger.info("Downloading/Loading Weights...")
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
    model.to_empty(device=device)
    
    try:
        load_model(model, policy_path, strict=True)
    except Exception:
        logger.info("Strict load failed (likely tied weights), performing manual tie...")
        state_dict = load_file(policy_path)
        model.load_state_dict(state_dict, strict=False)
        vlm = model.paligemma_with_expert.paligemma
        vlm.get_input_embeddings().weight.data = vlm.get_output_embeddings().weight.data.clone()
        
    model.to(device)
    model.eval()
    return model, model_config

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Serving on {device}...")
    
    model, model_config = load_so100_model(device)
    
    # Load Stats
    stats_path = "dataset_stats.json"
    if os.path.exists(stats_path):
        import json
        with open(stats_path, "r") as f:
            raw_stats = json.load(f)
        
        # Wrap into NormStats for openpi transforms
        # We need to map qpos_mean -> state, action_mean -> actions
        state_stats = _normalize.NormStats(
            mean=np.array(raw_stats.get("qpos_mean", np.zeros(14))),
            std=np.array(raw_stats.get("qpos_std", np.ones(14)))
        )
        action_stats = _normalize.NormStats(
            mean=np.array(raw_stats.get("action_mean", np.zeros(14))),
            std=np.array(raw_stats.get("action_std", np.ones(14)))
        )
        norm_stats = {"state": state_stats, "actions": action_stats}
    else:
        logger.warning("dataset_stats.json not found! Using raw values.")
        norm_stats = None

    # Define Transforms
    tokenizer = _tokenizer.PaligemmaTokenizer(model_config.max_token_len)
    
    input_transforms = [
        _transforms.InjectDefaultPrompt("pour a cup of coffee"),
        SO100InputTransform(), # Remap observation keys
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(tokenizer),
        _transforms.Normalize(norm_stats),
    ]
    
    output_transforms = [
        _transforms.Unnormalize(norm_stats),
        SO100OutputRemap(), # 14-dim -> 6-DOF
    ]
    
    def log_infer_start(obs):
        logger.info("Starting model inference...")
        return model.sample_actions(device, obs, **policy._sample_kwargs)

    policy = _policy.Policy(
        model=model,
        transforms=input_transforms,
        output_transforms=output_transforms,
        is_pytorch=True,
        pytorch_device=device
    )
    # Monkey-patch to log
    original_sample = policy._sample_actions
    def patched_sample(*args, **kwargs):
        logger.info("Inside sample_actions...")
        res = original_sample(*args, **kwargs)
        logger.info("Finished sample_actions.")
        return res
    policy._sample_actions = patched_sample

    logger.info("Starting SO-100 WebSocket Server on port 8000...")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=8000
    )
    server.serve_forever()

if __name__ == "__main__":
    main()
