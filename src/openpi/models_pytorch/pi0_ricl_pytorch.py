
import logging
import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

# Re-implement Gemma Config to avoid importing flax-dependent code
@dataclass
class LoRAConfig:
    rank: int
    alpha: float

@dataclass
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: Dict[str, LoRAConfig] = field(default_factory=dict)

def get_gemma_config(variant: str) -> GemmaConfig:
    if variant == "dummy":
        return GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant == "gemma_300m":
        return GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return GemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b_lora":
        return GemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256,
                           lora_configs={"attn": LoRAConfig(16, 16.0), "ffn": LoRAConfig(16, 16.0)})
    if variant == "gemma_300m_lora":
        return GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256,
                           lora_configs={"attn": LoRAConfig(32, 32.0), "ffn": LoRAConfig(32, 32.0)})
    if variant == "gemma3_4b":
        return GemmaConfig(width=2304, depth=26, mlp_dim=9216, num_heads=8, num_kv_heads=4, head_dim=256)
    if variant == "gemma3_4b_lora":
        return GemmaConfig(width=2304, depth=26, mlp_dim=9216, num_heads=8, num_kv_heads=4, head_dim=256,
                           lora_configs={"attn": LoRAConfig(16, 16.0), "ffn": LoRAConfig(16, 16.0)})
    if variant == "gemma_860m":
        return GemmaConfig(width=1280, depth=26, mlp_dim=5120, num_heads=8, num_kv_heads=4, head_dim=256)
    if variant == "gemma_860m_lora":
         return GemmaConfig(width=1280, depth=26, mlp_dim=5120, num_heads=8, num_kv_heads=4, head_dim=256,
                            lora_configs={"attn": LoRAConfig(16, 16.0), "ffn": LoRAConfig(16, 16.0)})
    if variant == "gemma_670m":
        return GemmaConfig(width=1280, depth=20, mlp_dim=5120, num_heads=10, num_kv_heads=2, head_dim=256)
    if variant == "gemma_670m_lora":
        return GemmaConfig(width=1280, depth=20, mlp_dim=5120, num_heads=10, num_kv_heads=2, head_dim=256,
                           lora_configs={"attn": LoRAConfig(16, 16.0), "ffn": LoRAConfig(16, 16.0)})
    raise ValueError(f"Unknown variant: {variant}")


from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

def get_safe_dtype(target_dtype, device_type):
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype

def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    if dimension % 2 != 0: raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1: raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    if isinstance(device, str): device = torch.device(device)
    dtype = get_safe_dtype(torch.float64, device.type)
    
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

def make_att_2d_masks(pad_masks, att_masks):
    # Fix: ensure float/bool compatibility if needed, but torch allows bool ops
    if att_masks.ndim != 2: raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2: raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks

def make_ricl_attn_mask(list_of_attn_masks, batch_size, seq_len):
    device = list_of_attn_masks[0].device
    full_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
    current_idx = 0
    for mask in list_of_attn_masks:
        L = mask.shape[1]
        cumsum = torch.cumsum(mask, dim=1)
        block_mask = cumsum[:, None, :] <= cumsum[:, :, None]
        full_mask[:, current_idx:current_idx+L, current_idx:current_idx+L] = block_mask
        current_idx += L
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    final_mask = full_mask | causal_mask.unsqueeze(0)
    return final_mask

class PI0RiclPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.num_retrieved_observations = config.num_retrieved_observations

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.gradient_checkpointing_enabled = False

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
        return func(*args, **kwargs)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        if lang_tokens is not None:
             def lang_embed_func(lang_tokens):
                 lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
                 lang_emb_dim = lang_emb.shape[-1]
                 return lang_emb * math.sqrt(lang_emb_dim)
             
             lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
             embs.append(lang_emb)
             pad_masks.append(lang_masks)
             num_lang_embs = lang_emb.shape[1]
             att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # Fix: att_masks to tensor
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if state is not None:
                if self.state_proj.weight.dtype == torch.float32:
                    state = state.to(torch.float32)
                state_emb = self._apply_checkpoint(self.state_proj, state)
                embs.append(state_emb[:, None, :])
                bsize = state_emb.shape[0]
                device = state_emb.device
                state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
                pad_masks.append(state_mask)
                att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)
        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x) 
                return self.action_time_mlp_out(x)
            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
             def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)
             time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
             action_time_emb = action_emb
             adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def extract_single_observation(self, ricl_observation, prefix, train=False):
        def get_item(key):
            if isinstance(ricl_observation, dict): return ricl_observation.get(key)
            return getattr(ricl_observation, key, None)
        def get_nested(base, key):
             if base is None: return None
             if isinstance(base, dict): return base.get(key)
             return getattr(base, key, None)

        image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        prefix_images = get_item(f"{prefix}images")
        prefix_masks = get_item(f"{prefix}image_masks")
        
        images = []
        img_masks = []
        for key in image_keys:
            if prefix_images and get_nested(prefix_images, key) is not None:
                images.append(get_nested(prefix_images, key))
                m = get_nested(prefix_masks, key)
                if m is None:
                    if len(images) > 0:
                        B = images[-1].shape[0]
                        device = images[-1].device
                    else:
                        B = 1
                        device = "cpu"
                    m = torch.ones(B, dtype=torch.bool, device=device)
                img_masks.append(m)
        
        prompt_prefix = get_item(f"{prefix}tokenized_prompt_prefix")
        prompt_postfix = get_item(f"{prefix}tokenized_prompt_postfix")
        
        if prompt_postfix is not None and prompt_prefix is not None:
            lang_tokens = torch.cat([prompt_prefix, prompt_postfix], dim=1)
        elif prompt_prefix is not None:
            lang_tokens = prompt_prefix
        else:
            lang_tokens = None
            
        lang_masks = get_item(f"{prefix}tokenized_prompt_mask")
        state = get_item(f"{prefix}state")
        return images, img_masks, lang_tokens, lang_masks, state

    def forward(self, ricl_observation, actions, noise=None, time=None) -> Tensor:
        num_observations = self.num_retrieved_observations + 1
        all_embs = []
        all_pad_masks = []
        all_att_masks = []
        state = None
        
        for i in range(num_observations):
            prefix = f"retrieved_{i}_" if i < self.num_retrieved_observations else "query_"
            images, img_masks, lang_tokens, lang_masks, obs_state = self.extract_single_observation(ricl_observation, prefix, train=True)
            if i == num_observations - 1: state = obs_state
            p_embs, p_pad_masks, p_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            all_embs.append(p_embs)
            all_pad_masks.append(p_pad_masks)
            all_att_masks.append(p_att_masks)
            
        if noise is None: noise = self.sample_noise(actions.shape, actions.device)
        if time is None: time = self.sample_time(actions.shape[0], actions.device)
            
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        prefix_part = torch.cat(all_embs, dim=1)
        suffix_part = suffix_embs
        all_att_masks.append(suffix_att_masks)
        
        full_len = prefix_part.shape[1] + suffix_part.shape[1]
        full_att_2d_masks = make_ricl_attn_mask(all_att_masks, prefix_part.shape[0], full_len)
        prefix_pad_total = torch.cat(all_pad_masks, dim=1)
        full_pad_masks = torch.cat([prefix_pad_total, suffix_pad_masks], dim=1)
        full_pad_2d_masks = full_pad_masks[:, None, :] * full_pad_masks[:, :, None]
        final_att_mask = full_att_2d_masks & full_pad_2d_masks
        final_att_mask_4d = self._prepare_attention_masks_4d(final_att_mask)
        position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
        
        def forward_func_split(prefix, suffix, att_mask, pos_ids, cond):
            (_, output), _ = self.paligemma_with_expert.forward(
                attention_mask=att_mask,
                position_ids=pos_ids,
                inputs_embeds=[prefix, suffix],
                adarms_cond=[None, cond],
                use_cache=False
            )
            return output
             
        suffix_out = self._apply_checkpoint(
            forward_func_split, prefix_part, suffix_part, final_att_mask_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, ricl_observation, noise=None, num_steps=10) -> Tensor:
        bsize = ricl_observation.get("query_state").shape[0] if isinstance(ricl_observation, dict) else ricl_observation.query_state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
            
        num_observations = self.num_retrieved_observations + 1
        all_embs = []
        all_pad_masks = []
        all_att_masks = []
        query_state = None
        
        for i in range(num_observations):
            prefix = f"retrieved_{i}_" if i < self.num_retrieved_observations else "query_"
            images, img_masks, lang_tokens, lang_masks, state = self.extract_single_observation(ricl_observation, prefix, train=False)
            p_embs, p_pad_masks, p_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            all_embs.append(p_embs)
            all_pad_masks.append(p_pad_masks)
            all_att_masks.append(p_att_masks)
            if i == num_observations - 1: query_state = state

        prefix_embs = torch.cat(all_embs, dim=1)
        prefix_pad_masks = torch.cat(all_pad_masks, dim=1)
        prefix_att_2d_masks = make_ricl_attn_mask(all_att_masks, prefix_embs.shape[0], prefix_embs.shape[1])
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
        prefix_att_2d_masks = prefix_att_2d_masks & prefix_pad_2d_masks
        
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None],
            use_cache=True
        )
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step_ricl(query_state, prefix_pad_masks, all_att_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step_ricl(self, state, prefix_pad_masks, prefix_att_masks_list, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_att_masks_list = [suffix_att_masks]
        full_att_masks_list = prefix_att_masks_list + suffix_att_masks_list
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        total_len = prefix_len + suffix_len
        batch_size = prefix_pad_masks.shape[0]
        full_ricl_mask = make_ricl_attn_mask(full_att_masks_list, batch_size, total_len)
        full_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        full_pad_2d = full_pad_masks[:, None, :] * full_pad_masks[:, :, None]
        full_mask = full_ricl_mask & full_pad_2d
        step_mask = full_mask[:, prefix_len:, :]
        step_mask_4d = self._prepare_attention_masks_4d(step_mask)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=step_mask_4d, position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond]
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
