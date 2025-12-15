"""
Map measured CapTrak positions to REVE standard positions using spherical coordinates.
This creates a standard set of position embeddings that can be reused across subjects.

METHOD: Uses spherical coordinates (theta, phi) to avoid scale issues.
- CapTrak positions: Already has theta/phi in electrode_positions.json
- REVE positions: Converted from Cartesian (x, y, z) to spherical coordinates
- Matching: Done using angular distance (scale-independent)

DATA SOURCES:
- Measured positions: Your CapTrak 3D digitized coordinates from XDF files (electrode_positions.json)
- Standard positions: REVE's pre-trained embeddings from brain-bzh/reve-positions model (reve_all_positions.json)
"""

import json
from pathlib import Path
import numpy as np


def cartesian_to_spherical(x, y, z):
    """Convert Cartesian (x, y, z) to spherical (r, theta, phi) coordinates.
    theta: azimuth (0-360 degrees, measured from x-axis in xy-plane)
    phi: elevation (0-180 degrees, from z-axis)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x) * 180 / np.pi  # -180 to 180
    phi = np.arccos(z / r) * 180 / np.pi if r > 0 else 0  # 0 to 180
    return theta, phi


def load_captrak_positions():
    """Load your measured 96 electrode positions (converted to spherical coords)."""
    json_path = Path(__file__).parent.parent / "electrodes_pos" / "electrode_positions.json"
    
    with open(json_path, 'r') as f:
        electrodes = json.load(f)
    
    # Convert Cartesian x,y,z to proper spherical coordinates
    # (CapTrak's stored theta/phi are in a different convention, so we recompute)
    positions = {}
    for name in sorted([int(k) for k in electrodes.keys()]):
        el = electrodes[str(name)]
        theta, phi = cartesian_to_spherical(el['x'], el['y'], el['z'])
        positions[name] = np.array([theta, phi])
    
    return positions


def get_reve_standard_positions():
    """
    Load ALL positions from REVE's pre-trained embeddings and convert to spherical coords.
    This includes standard 10-20 plus extended electrode systems.
    REVE knows 543 total positions, so we use all of them for maximum coverage.
    """
    
    json_path = Path(__file__).parent.parent / "electrodes_pos" / "reve_all_positions.json"
    
    with open(json_path, 'r') as f:
        all_reve_positions = json.load(f)
    
    # Convert all Cartesian positions to spherical coordinates (theta, phi)
    standard_positions = {}
    for pos_name, coords in all_reve_positions.items():
        theta, phi = cartesian_to_spherical(coords['x'], coords['y'], coords['z'])
        standard_positions[pos_name] = np.array([theta, phi])
    
    return standard_positions


def angular_distance(theta1_phi1, theta2_phi2):
    """
    Calculate angular distance between two points on a sphere using spherical coordinates.
    Uses the haversine formula for accurate great-circle distance.
    theta, phi in degrees
    """
    theta1, phi1 = theta1_phi1
    theta2, phi2 = theta2_phi2
    
    # Convert to radians
    theta1_rad = theta1 * np.pi / 180
    phi1_rad = phi1 * np.pi / 180
    theta2_rad = theta2 * np.pi / 180
    phi2_rad = phi2 * np.pi / 180
    
    # Haversine formula (angular distance on unit sphere)
    dlat = phi2_rad - phi1_rad
    dlon = theta2_rad - theta1_rad
    a = np.sin(dlat/2)**2 + np.cos(phi1_rad) * np.cos(phi2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return c * 180 / np.pi  # Return in degrees


def find_closest_standard_position(measured_pos, standard_positions):
    """Find the closest standard position for a measured electrode using angular distance."""
    min_distance = float('inf')
    closest_name = None
    
    for std_name, std_pos in standard_positions.items():
        distance = angular_distance(measured_pos, std_pos)
        if distance < min_distance:
            min_distance = distance
            closest_name = std_name
    
    return closest_name, min_distance


def create_mapping():
    """Map all 96 measured electrodes to REVE standard positions."""
    
    print("=" * 80)
    print("MAPPING YOUR 96 ELECTRODES TO REVE STANDARD POSITIONS")
    print("=" * 80 + "\n")
    
    measured = load_captrak_positions()
    standard = get_reve_standard_positions()
    
    print(f"REVE standard positions: {len(standard)}")
    print(f"Your measured electrodes: {len(measured)}\n")
    
    # Create mapping
    mapping = {}
    distances = []
    
    for electrode_num in sorted(measured.keys()):
        measured_pos = measured[electrode_num]
        closest_name, distance = find_closest_standard_position(measured_pos, standard)
        
        mapping[electrode_num] = {
            'standard_position': closest_name,
            'distance_degrees': float(distance),
            'measured_pos': measured_pos.tolist(),
            'standard_pos': standard[closest_name].tolist()
        }
        distances.append(distance)
    
    # Statistics
    print(f"Mapping Statistics:")
    print(f"  Mean distance: {np.mean(distances):.1f}°")
    print(f"  Max distance: {np.max(distances):.1f}°")
    print(f"  Min distance: {np.min(distances):.1f}°\n")
    
    # Show sample mappings
    print("Sample mappings (first 10):")
    print(f"{'Electrode':<12} {'→ Standard':<15} {'Distance (°)':<15}")
    print("-" * 42)
    for i, (num, data) in enumerate(list(mapping.items())[:10]):
        print(f"{num:<12} {data['standard_position']:<15} {data['distance_degrees']:<15.1f}")
    
    print("\n...")
    
    # Save mapping
    output_path = Path(__file__).parent.parent / "electrodes_pos" / "electrode_mapping_to_standard.json"
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✓ Mapping saved to: {output_path.name}\n")
    
    # Show REVE positions
    unique_standards = set([data['standard_position'] for data in mapping.values()])
    
    print("=" * 80)
    print(f"MAPPING TO ALL {len(unique_standards)} REVE POSITIONS:")
    print("=" * 80)
    
    print(f"\nYour 96 electrodes map to {len(unique_standards)} unique REVE positions\n")
    print(f"(REVE knows {len(standard)} total positions, so you're using {len(unique_standards)} of them)\n")
    print(sorted(unique_standards))
    
    # Code to use
    print("\n\nCode to get REVE embeddings:\n")
    print("```python")
    print(f"pos_bank = AutoModel.from_pretrained('brain-bzh/reve-positions', trust_remote_code=True)")
    print(f"standard_positions = {sorted(unique_standards)}")
    print(f"positions = pos_bank(standard_positions)  # Shape: ({len(unique_standards)}, embedding_dim)")
    print("```")
    
    return mapping


if __name__ == "__main__":
    mapping = create_mapping()
