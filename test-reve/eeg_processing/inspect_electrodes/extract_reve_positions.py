"""
Extract actual position coordinates from REVE's model.
"""

from transformers import AutoModel
import numpy as np
import json
from pathlib import Path

# Load REVE's position bank
pos_bank = AutoModel.from_pretrained('brain-bzh/reve-positions', trust_remote_code=True)

# Get all position names
all_positions = pos_bank.position_names
print(f"Total positions in REVE: {len(all_positions)}")

# Get the position embeddings (3D coordinates)
# REVE position embeddings are 3D coordinates
position_embeddings = pos_bank.embedding.detach().cpu().numpy()

print(f"Embedding shape: {position_embeddings.shape}")
print(f"Expected: ({len(all_positions)}, 3)")

# Save all positions with their coordinates
positions_dict = {}
for i, pos_name in enumerate(all_positions):
    coord = position_embeddings[i]
    positions_dict[pos_name] = {
        'x': float(coord[0]),
        'y': float(coord[1]),
        'z': float(coord[2])
    }

# Save to JSON
output_path = Path(__file__).parent.parent / "electrodes_pos" / "reve_all_positions.json"
with open(output_path, 'w') as f:
    json.dump(positions_dict, f, indent=2)

print(f"\n✓ Saved all {len(positions_dict)} REVE positions to {output_path.name}")

# Show a sample
print("\nSample REVE positions (first 20):")
print(f"{'Position':<20} {'X':<12} {'Y':<12} {'Z':<12}")
print("-" * 56)
for i, (name, coords) in enumerate(list(positions_dict.items())[:20]):
    print(f"{name:<20} {coords['x']:<12.2f} {coords['y']:<12.2f} {coords['z']:<12.2f}")

# Show standard 10-20 positions specifically
standard_10_20 = ['Fp1', 'Fp2', 'Fpz', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                  'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                  'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                  'O1', 'O2', 'Oz']

print("\n" + "="*60)
print("STANDARD 10-20 POSITIONS IN REVE:")
print("="*60 + "\n")
print(f"{'Position':<20} {'X':<12} {'Y':<12} {'Z':<12} {'Available':<12}")
print("-" * 70)
for name in standard_10_20:
    if name in positions_dict:
        coords = positions_dict[name]
        print(f"{name:<20} {coords['x']:<12.2f} {coords['y']:<12.2f} {coords['z']:<12.2f} {'✓':<12}")
    else:
        print(f"{name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'✗':<12}")
