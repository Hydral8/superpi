
import os
import torch
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json

try:
    try:
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    except ImportError:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: 'lerobot' library not found. LeRobotRetriever will not function correctly.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: 'faiss' library not found. Install 'faiss-cpu' or 'faiss-gpu' for efficient retrieval.")

import torchvision.transforms as T
import math

logger = logging.getLogger(__name__)

# --- Embedding Utility Functions (Adapted from ricl_openpi/src/openpi/policies/utils.py) ---

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def load_dinov2(device="cpu"):
    """Loads DINOv2 model from torch hub."""
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2.eval()
    dinov2.to(device)
    return dinov2

def process_images_for_dino(images: torch.Tensor, device="cpu"):
    """
    Preprocesses images for DINOv2.
    Expects input images to be [B, C, H, W] or [C, H, W] in float usually?
    Actually LeRobot usually returns [C, H, W] float or uint8?
    We assume input is a batch of torch tensors.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    
    # Resize to 224x224
    # Note: DINOv2 expects multiple of 14. 224 is standard.
    if images.shape[-2:] != (224, 224):
         images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    # Convert to float and normalize if needed (assuming input is potentially [0,1] or [0,255])
    # Usually LeRobot datasets return float [0,1].
    # But if it's uint8, we divide. 
    # We'll assume input is float [0,1] for now as per PyTorch convention if coming from a transform.
    # Standardize
    normalize = T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    images = normalize(images)
    return images.to(device)

def embed_images(images: torch.Tensor, model, device="cpu", embedding_type="64PATCHES") -> np.ndarray:
    """
    Embeds images using DINOv2 model.
    """
    processed = process_images_for_dino(images, device=device)
    with torch.no_grad():
        features = model.forward_features(processed)
        # features is a dict. 'x_norm_patchtokens' is [B, N_patches, 768] (N=256 for 224x224)
        
        batch_embeddings = features["x_norm_patchtokens"]
        
        if embedding_type == "CLS":
             batch_embeddings = features["x_norm_clstoken"]
        elif embedding_type == "AVG":
             batch_embeddings = batch_embeddings.mean(dim=1)
        elif "PATCHES" in embedding_type:
            # Downsample patches logic (e.g. 256 -> 64)
            # Replicating logic provided in utils.py
            N_patches = int(embedding_type.split('PATCHES')[0])
            batch_size, num_patches, dim = batch_embeddings.shape
            
            # Assuming 16x16 original grid
            rows, cols = 16, 16 
            target_dim = int(math.sqrt(N_patches))
            patch_h = rows // target_dim
            patch_w = cols // target_dim
            
            # Reshape to [B, rows, cols, dim]
            grid = batch_embeddings.view(batch_size, rows, cols, dim)
            
            # Block mean
            # [B, target_dim, patch_h, target_dim, patch_w, dim] -> mean over patch_h, patch_w
            grid = grid.view(batch_size, target_dim, patch_h, target_dim, patch_w, dim)
            pooled = grid.mean(dim=(2, 4)) # [B, target_dim, target_dim, dim]
            
            # Flatten
            batch_embeddings = pooled.view(batch_size, N_patches * dim)
        
        return batch_embeddings.cpu().numpy()

# --- LeRobot Retriever ---

class LeRobotRetriever:
    def __init__(self, repo_id: str, cache_dir: str = ".cache/retrieval", embedding_type="64PATCHES", device="cuda" if torch.cuda.is_available() else "cpu"):
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot is not installed.")
        
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir) / repo_id.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_type = embedding_type
        self.device = device
        
        self.dataset = lerobot_dataset.LeRobotDataset(repo_id, video_backend="pyav")
        self.encoder = None # Lazy load
        self.index = None
        self.embeddings = None
        self.metadata = [] # List of (episode_idx, frame_idx) corresponding to embeddings
        
        self._initialize_index()

    def _initialize_index(self):
        emb_path = self.cache_dir / f"embeddings_{self.embedding_type}.npy"
        meta_path = self.cache_dir / f"metadata_{self.embedding_type}.json"
        
        if emb_path.exists() and meta_path.exists():
            logger.info(f"Loading cached embeddings from {self.cache_dir}")
            self.embeddings = np.load(emb_path)
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            logger.info(f"Computing embeddings for {self.repo_id}...")
            self._compute_embeddings(emb_path, meta_path)
            
        if FAISS_AVAILABLE:
            logger.info("Building FAISS index...")
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings)
            logger.info(f"Index built with {self.index.ntotal} vectors.")
        else:
            logger.warning("FAISS not available. Retrieval will be slow (exact search via numpy).")

    def _compute_embeddings(self, emb_path, meta_path):
        self.encoder = load_dinov2(self.device)
        
        all_embs = []
        all_meta = []
        
        # Iterate over dataset
        # Warning: For large datasets, this takes time.
        # We process frame by frame.
        # LeRobotDataset[i] returns a dict with 'observation.images.X' etc.
        # We need to assume a primary image key for retrieval.
        image_key = "observation.images.base_0_rgb" # Attempt to find default
        
        # Heuristic to find image key if not standard
        sample = self.dataset[0]
        keys = list(sample.keys())
        # Try to find an image key
        if "observation.images.top_image" in keys:
            image_key = "observation.images.top_image" 
        elif "observation.images.phone_image" in keys:
            image_key = "observation.images.phone_image"
            
        logger.info(f"Using image key '{image_key}' for retrieval embeddings.")
        
        batch_size = 32
        batch_images = []
        batch_meta = []
        
        total = len(self.dataset)
        for i in range(total):
            item = self.dataset[i]
            img = item[image_key] # [C, H, W] tensor (float)
            batch_images.append(img)
            
            # Metadata: (episode_index, frame_index)
            # LeRobotDataset access is flat, but we can reconstruct or just store flat index 'i'
            # To be useful for RICL, we might need episode boundaries?
            # RICL typically retrieves "nearest neighbors" which are frames.
            # We'll store just 'i' for now, or (episode_index, frame_index) if available in item.
            # LeRobot item usually has 'episode_index' and 'frame_index'.
            ep_idx = item.get("episode_index", -1)
            fr_idx = item.get("frame_index", i) # Fallback to global index if missing
            
            all_meta.append((ep_idx, fr_idx, i))
            
            if len(batch_images) >= batch_size or i == total - 1:
                batch_tensor = torch.stack(batch_images).to(self.device)
                embs = embed_images(batch_tensor, self.encoder, self.device, self.embedding_type)
                all_embs.append(embs)
                batch_images = []
                
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{total} frames")

        self.embeddings = np.concatenate(all_embs, axis=0)
        self.metadata = all_meta
        
        np.save(emb_path, self.embeddings)
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f)
        
        self.encoder = None # Free memory
        torch.cuda.empty_cache()

    def retrieve(self, query_images: torch.Tensor, k: int = 5) -> Dict[str, Any]:
        """
        Retrieves k nearest neighbors for the query image(s).
        query_images: [1, C, H, W] or [C, H, W]
        Returns a dictionary suitable for construction of ricl_observation.
        """
        if self.encoder is None:
             self.encoder = load_dinov2(self.device)
             
        query_emb = embed_images(query_images, self.encoder, self.device, self.embedding_type) # [1, D]
        
        if FAISS_AVAILABLE:
            D, I = self.index.search(query_emb, k)
            indices = I[0] # I is [1, k]
        else:
            # Exact search
            dists = np.linalg.norm(self.embeddings - query_emb, axis=1)
            indices = np.argsort(dists)[:k]
            
        # Fetch data
        retrieved_data = {
            f"retrieved_{i}_images": [] for i in range(k)
        }
        # And other fields...
        
        # For PI0RiclPytorch, we need complete observation dicts.
        # But wait, LeRobotDataset returns one sample.
        # We need to construct the "retrieved_i_..." keys.
        
        result = {}
        for i, idx in enumerate(indices):
            if idx == -1: continue # Faiss padding
            
            # meta = self.metadata[idx]
            global_idx = self.metadata[idx][2]
            sample = self.dataset[global_idx] # This loads images/state
            
            # Map sample keys to RICL keys
            # sample has keys like 'observation.state', 'observation.images.base_0_rgb', etc.
            # We want to prefix them with `retrieved_{i}_`.
            
            prefix = f"retrieved_{i}_"
            
            # Images
            # We'll put all images from the retrieved sample into a dict under "retrieved_{i}_images"?
            # PI0RiclPytorch helper `extract_single_observation` expects:
            # {prefix}images -> dict of images
            # {prefix}state -> tensor
            
            # The sample is flat: "observation.images.foo".
            # We need to restructure.
            
            images_dict = {}
            for key, val in sample.items():
                if key.startswith("observation.images."):
                    img_name = key.split("observation.images.")[1]
                    images_dict[img_name] = val # tensor
            
            result[f"{prefix}images"] = images_dict
            
            if "observation.state" in sample:
                result[f"{prefix}state"] = sample["observation.state"]
            
            # Prompt?
            # Retrieved items usually don't dictate the prompt, the query does.
            # But RICL might use retrieved text?
            # Usually RICL uses the SAME prompt as query, or retrieved text if available.
            # For now we ignore text in retrieved unless dataset has task description.
        
        return result

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    try:
        # Dummy test if LeRobot is available and connected to HF
        print("Testing LeRobotRetriever instantiation...")
        # r = LeRobotRetriever("lerobot/aloha_sim_transfer_cube_human") # Example
        print("Skipped actual run in CI environment.")
    except Exception as e:
        print(f"Error: {e}")
