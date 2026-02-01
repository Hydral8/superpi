import os
import sys
import logging
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model, load_file

# Add src to path
sys.path.insert(0, "lib")
sys.path.insert(0, "src")

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.models import model as _model
from openpi.models import tokenizer

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

class SO100Adapter:
    def __init__(self, stats_path="dataset_stats.json"):
        # We only care about Right Arm (Indices 7-13) based on your JSON
        self.start_idx = 7
        self.end_idx = 14
        
        # --- LOAD STATS ---
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.action_mean = np.array(stats.get("qpos_mean", np.zeros(14)))
            self.action_std = np.array(stats.get("qpos_std", np.ones(14)))
            self.action_std = np.where(self.action_std < 1e-5, 1.0, self.action_std) # Avoid div/0
            logger.info("✅ Loaded Normalization Statistics.")
        else:
            logger.warning("⚠️ No stats found! Using raw outputs (Dangerous).")
            self.action_mean = np.zeros(14)
            self.action_std = np.ones(14)

        # SO-100 Hardware Calibration (Adjust signs if motors move backwards)
        self.joint_signs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.joint_offsets = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def process_actions(self, raw_actions):
        # 1. Denormalize (Z-Score -> Radians)
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.float().cpu().numpy()
        denorm = (raw_actions * self.action_std) + self.action_mean
        
        if len(denorm.shape) == 3: denorm = denorm[0]

        # 2. Extract Right Arm (Indices 7-13)
        right_arm = denorm[:, self.start_idx:self.end_idx] 
        
        # 3. Drop Forearm Roll (Index 3 in the 7-dim block)
        # ALOHA Right: [Waist, Shoulder, Elbow, Roll(DROP), Pitch, Roll, Grip]
        # Keep:        [0,     1,        2,               4,     5,    6   ]
        so100_raw = right_arm[:, [0, 1, 2, 4, 5, 6]]
        
        # 4. Apply Calibration
        return (so100_raw * self.joint_signs) + self.joint_offsets

def load_image_tensor(image_path, device):
    """Loads and preprocesses an image for the model."""
    if not os.path.exists(image_path):
        logger.warning(f"Image {image_path} not found! Using noise.")
        return torch.randn((1, 3, 224, 224), device=device)
    
    try:
        img = Image.open(image_path).convert("RGB")
        # Standard Transforms: Resize -> Tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device).to(torch.bfloat16)  # Match model dtype if needed
        logger.info(f"Loaded image: {image_path}")
        return img_tensor
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return torch.randn((1, 3, 224, 224), device=device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = "exla-ai/openpie-0.6"
    
    # 1. Download & Load Model
    logger.info("Loading Model...")
    policy_path = hf_hub_download(repo_id=repo_id, filename="policy.safetensors")
    
    model_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_2b",
        action_dim=14, # Matches your JSON 'observation.state' shape
        action_horizon=50,
        dtype="bfloat16"
    )
    
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = pi0_pytorch.PI0Pytorch(model_config)
    model.to_empty(device=device)
    
    # Robust Weight Loading
    try:
        load_model(model, policy_path, strict=True)
    except Exception:
        # Fallback for embedding mismatch
        state_dict = load_file(policy_path)
        model.load_state_dict(state_dict, strict=False)
        vlm = model.paligemma_with_expert.paligemma
        vlm.get_input_embeddings().weight.data = vlm.get_output_embeddings().weight.data.clone()
        
    model.to(device)
    model.eval()
    
    # 2. Prepare Inputs
    adapter = SO100Adapter()
    prompt = "pour a cup of coffee"
    image_path = "scripts/image_6cfd92.jpg" # Your uploaded image

    # Load the Real Image (Keep this for Base)
    real_img_tensor = load_image_tensor(image_path, device)

    # Create a "Blackout" tensor for the wrist
    # (This tells the model "I can't see anything from the wrist", reducing conflict)
    black_tensor = torch.zeros_like(real_img_tensor)

    images = {
        "base_0_rgb": real_img_tensor,       # GOOD: The model sees the cup here
        "left_wrist_0_rgb": black_tensor,    # MASKED: Ignore left arm
        "right_wrist_0_rgb": black_tensor    # MASKED: Stop the hallucination
    }

    image_masks = {k: torch.ones((1,), device=device, dtype=torch.bool) for k in images}
    
    # Set current state to 'Mean' (Home) to avoid jumpiness
    state = torch.from_numpy(adapter.action_mean).unsqueeze(0).to(device).float()
    
    tok = tokenizer.PaligemmaTokenizer(model_config.max_token_len)
    tokenized, mask = tok.tokenize(prompt, state=state.cpu().numpy()[0])
    
    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=torch.from_numpy(tokenized).to(device).unsqueeze(0),
        tokenized_prompt_mask=torch.from_numpy(mask).to(device).unsqueeze(0)
    )
    
    # 3. Run Inference
    logger.info("Generating Actions...")
    with torch.no_grad():
        actions = model.sample_actions(device, obs, num_steps=20)
        
    # 4. View Results
    so100_actions = adapter.process_actions(actions)
    
    print("\n--- SO-100 TRAJECTORY (First 5 Steps) ---")
    print("Format: [Waist, Shoulder, Elbow, Pitch, Roll, Gripper]")
    for i in range(min(5, len(so100_actions))):
        print(f"Step {i}: {so100_actions[i].round(4)}")

if __name__ == "__main__":
    main()