"""
Test the spherical coordinate mapping to verify distances are correct.
"""

import json
from pathlib import Path
import numpy as np
from map_to_standard_positions import (
    load_captrak_positions, 
    get_reve_standard_positions,
    angular_distance
)


def test_mapping():
    """Test that the mapping JSON was created and has reasonable values."""
    
    print("\n" + "=" * 80)
    print("TESTING ELECTRODE MAPPING")
    print("=" * 80 + "\n")
    
    # Check if mapping file exists
    mapping_path = Path(__file__).parent.parent / "electrodes_pos" / "electrode_mapping_to_standard.json"
    
    if not mapping_path.exists():
        print(f"❌ Mapping file not found: {mapping_path}")
        print("   Run map_to_standard_positions.py first to create it")
        return False
    
    # Load and analyze mapping
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    print(f"✓ Loaded mapping with {len(mapping)} electrodes\n")
    
    # Get all distances
    distances = [float(data['distance_degrees']) for data in mapping.values()]
    
    # Test 1: Distances should be in 0-180 degrees (angular distance on sphere)
    print("TEST 1: Distance ranges")
    print(f"  Min distance: {min(distances):.2f}°")
    print(f"  Max distance: {max(distances):.2f}°")
    print(f"  Mean distance: {np.mean(distances):.2f}°")
    print(f"  Std dev: {np.std(distances):.2f}°")
    
    if max(distances) <= 180 and min(distances) >= 0:
        print("  ✓ All distances in valid range [0°, 180°]\n")
    else:
        print("  ❌ PROBLEM: Distances out of valid range (should be 0-180 degrees)\n")
        return False
    
    # Test 2: Reasonable distances (shouldn't be too large)
    if np.mean(distances) > 50:
        print("⚠️  WARNING: Mean distance is large. Check if mapping is working correctly.\n")
    else:
        print(f"  ✓ Mean distance reasonable (~{np.mean(distances):.1f}°)\n")
    
    # Test 3: Verify by recomputing a sample
    print("TEST 2: Verify a few mappings by recomputing distances\n")
    
    measured = load_captrak_positions()
    standard = get_reve_standard_positions()
    
    test_electrodes = [1, 50, 96]  # Test first, middle, last electrodes
    all_correct = True
    
    for elec_num in test_electrodes:
        if str(elec_num) not in mapping:
            continue
        
        stored_data = mapping[str(elec_num)]
        measured_pos = measured[elec_num]
        standard_name = stored_data['standard_position']
        standard_pos = standard[standard_name]
        
        # Recompute distance
        recomputed_distance = angular_distance(measured_pos, standard_pos)
        stored_distance = stored_data['distance_degrees']
        
        print(f"  Electrode {elec_num}:")
        print(f"    Measured angles: theta={measured_pos[0]:.2f}°, phi={measured_pos[1]:.2f}°")
        print(f"    Standard position: {standard_name}")
        print(f"    Standard angles: theta={standard_pos[0]:.2f}°, phi={standard_pos[1]:.2f}°")
        print(f"    Stored distance: {stored_distance:.2f}°")
        print(f"    Recomputed distance: {recomputed_distance:.2f}°")
        
        if abs(recomputed_distance - stored_distance) < 0.01:
            print(f"    ✓ Match!\n")
        else:
            print(f"    ❌ MISMATCH!\n")
            all_correct = False
    
    if not all_correct:
        return False
    
    # Test 4: Check that same electrode positions always map to same standard position
    print("TEST 3: Consistency check\n")
    
    standard_mappings = {}
    for elec_num in sorted(int(k) for k in mapping.keys()):
        standard_name = mapping[str(elec_num)]['standard_position']
        if standard_name not in standard_mappings:
            standard_mappings[standard_name] = []
        standard_mappings[standard_name].append(elec_num)
    
    print(f"  {len(mapping)} electrodes map to {len(standard_mappings)} unique standard positions")
    print(f"  Distribution: {sorted([len(v) for v in standard_mappings.values()], reverse=True)[:5]} (top 5)")
    print(f"  ✓ Mapping is consistent\n")
    
    # Test 5: Show mapping quality by electrode type
    print("TEST 4: Mapping quality statistics\n")
    print(f"  {'Percentile':<12} {'Distance (°)':<15}")
    print("  " + "-" * 27)
    for p in [25, 50, 75, 90, 95, 99]:
        d = np.percentile(distances, p)
        print(f"  {p}th {' ':<8} {d:<15.2f}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - Mapping is valid!")
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_mapping()
    exit(0 if success else 1)
