import json
import numpy as np

# --- SAFE STATISTICS GENERATOR ---
# Based on your meta/info.json structure (14 DOF)

# 1. MEANS (Home Position in Radians)
# We set these to 0.0 (Center). 
# If your robot's home is different (e.g., Elbow up), change index 9.
safe_means = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # Left Arm (Indices 0-6) - Ignored
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Right Arm (Indices 7-13)
]

# 2. STDS (Safety Scale)
# The model outputs Z-scores. We multiply them by this value to get Radians.
# 0.3 rads is approx 17 degrees. This is a safe starting speed.
# Increase to 0.5 or 0.8 if the arm moves too little.
scale = 0.3 
safe_stds = [scale] * 14 

stats = {
    "qpos_mean": safe_means,
    "qpos_std": safe_stds,
    "action_mean": safe_means, # Pi0 sometimes looks for this key
    "action_std": safe_stds
}

with open("dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

print(f"✅ Created 'dataset_stats.json' with scale factor {scale}.")
print("Indices 7-13 (Right Arm) are ready for SO-100.")