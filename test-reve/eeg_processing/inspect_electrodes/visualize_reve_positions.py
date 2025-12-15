"""
Visualize REVE position bank standards.
Shows different standard electrode systems (10-20, extended, etc.) separately.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import mne
import matplotlib.pyplot as plt


def load_reve_positions():
    """Load all REVE positions."""
    json_path = Path(__file__).parent.parent / "electrodes_pos" / "reve_all_positions.json"
    
    if json_path.exists():
        print(f"✓ Loading REVE positions from: {json_path.name}\n")
        with open(json_path, 'r') as f:
            return json.load(f)
    
    print("REVE positions file not found")
    return None


def extract_display_names(position_names):
    """
    Extract display names for positions.
    If all positions share the same prefix (like E1, E2, E3), show just numbers.
    If prefixes vary (like Fp1, Cz, Pz), show full names.
    For Biosemi, remove the 'biosemi128_' prefix first.
    """
    if not position_names:
        return position_names
    
    # First, remove 'biosemi128_' or similar prefixes
    cleaned_names = []
    for name in position_names:
        if 'biosemi' in name.lower():
            # Remove 'biosemi128_' or 'biosemi_' prefix
            cleaned = name.split('_', 1)[-1]  # Get everything after first underscore
            cleaned_names.append(cleaned)
        else:
            cleaned_names.append(name)
    
    # Find common prefix in cleaned names
    prefixes = set()
    for name in cleaned_names:
        # Extract alphabetic prefix
        prefix = ''.join([c for c in name if c.isalpha()])
        prefixes.add(prefix)
    
    # If all positions share one prefix, show just the numeric part
    if len(prefixes) == 1:
        return [
            ''.join([c for c in name if c.isdigit() or c == 'h'])  # Keep 'h' for half-way electrodes
            for name in cleaned_names
        ]
    else:
        # Different prefixes - keep cleaned names (with biosemi prefix removed)
        return cleaned_names


def categorize_positions(positions_dict):
    """
    Categorize REVE positions by standard system.
    Groups positions by common naming patterns.
    """
    categories = {
        '10-20 Extended (A-Z, 0-9)': [],
        'Biosemi': [],
        'Letter code E': [],
        'Letter codes (B, C, D, F)': [],
        'Special (M, I, P, O)': [],
        'Other': []
    }
    
    for pos_name in sorted(positions_dict.keys()):
        if any(x in pos_name for x in ['biosemi', 'Biosemi']):
            categories['Biosemi'].append(pos_name)
        elif pos_name[0] == 'E':
            categories['Letter code E'].append(pos_name)
        elif pos_name[0] in ['B', 'C', 'D', 'F']:
            categories['Letter codes (B, C, D, F)'].append(pos_name)
        elif pos_name[0] in ['M', 'I', 'P', 'O']:
            categories['Special (M, I, P, O)'].append(pos_name)
        elif len(pos_name) <= 3 and any(c.isdigit() for c in pos_name):
            categories['10-20 Extended (A-Z, 0-9)'].append(pos_name)
        else:
            categories['Other'].append(pos_name)
    
    return categories


def plot_reve_system(position_names, all_positions, system_name):
    """Plot one REVE electrode system."""
    if not position_names:
        return
    
    print(f"\n{'='*70}")
    print(f"Plotting {system_name}: {len(position_names)} electrodes")
    print(f"{'='*70}")
    
    # Get display names (simplified for uniform prefixes)
    display_names = extract_display_names(position_names)
    
    # Extract positions for this system
    positions = {}
    for orig_name, display_name in zip(position_names, display_names):
        coords = all_positions[orig_name]
        # REVE positions are already normalized to unit sphere
        x, y, z = coords['x'], coords['y'], coords['z']
        positions[display_name] = np.array([x, y, z])
    
    # Create MNE info
    ch_names = list(positions.keys())
    info = mne.create_info(ch_names, sfreq=500, ch_types='eeg')
    raw = mne.io.RawArray(np.zeros((len(ch_names), 1000)), info)
    
    # Create montage
    montage = mne.channels.make_dig_montage(
        ch_pos=positions,
        coord_frame='head'
    )
    
    raw.set_montage(montage)
    
    # Plot 3D
    print(f"Displaying 3D view with {len(ch_names)} electrodes...")
    fig_3d = mne.viz.plot_sensors(raw.info, kind='3d', 
                                   title=f'{system_name} - 3D View', 
                                   show_names=True)
    
    # Plot 2D (topomap)
    print(f"Displaying 2D view (topomap)...")
    fig_2d = mne.viz.plot_sensors(raw.info, kind='topomap', 
                                   title=f'{system_name} - 2D Layout', 
                                   show_names=True)
    
    plt.show()


def visualize_all_systems(all_positions):
    """Visualize each REVE standard system separately."""
    categories = categorize_positions(all_positions)
    
    print(f"\nFound {len(all_positions)} total REVE positions")
    print("\nBreakdown by category:")
    for category, positions in categories.items():
        if positions:
            print(f"  {category}: {len(positions)} positions")
    
    # Interactive menu
    while True:
        print(f"\n{'='*70}")
        print("REVE Position Systems Available:")
        print(f"{'='*70}")
        
        systems = [(name, pos) for name, pos in categories.items() if pos]
        for i, (name, positions) in enumerate(systems, 1):
            print(f"{i}. {name} ({len(positions)} electrodes)")
        
        print(f"{len(systems) + 1}. View all systems (one at a time)")
        print(f"{len(systems) + 2}. Exit")
        
        try:
            choice = input(f"\nSelect system to visualize (1-{len(systems) + 2}): ").strip()
            choice = int(choice)
            
            if choice < 1 or choice > len(systems) + 2:
                print("Invalid choice, try again")
                continue
            
            if choice == len(systems) + 2:
                print("Exiting...")
                break
            
            if choice == len(systems) + 1:
                # Show all one by one
                for name, positions in systems:
                    plot_reve_system(positions, all_positions, name)
            else:
                # Show selected system
                selected_idx = choice - 1
                system_name, positions = systems[selected_idx]
                plot_reve_system(positions, all_positions, system_name)
        
        except ValueError:
            print("Invalid input, please enter a number")


def main():
    print("\n" + "="*70)
    print("REVE Position Bank Visualization")
    print("="*70 + "\n")
    
    all_positions = load_reve_positions()
    
    if not all_positions:
        print("Could not load REVE positions")
        return
    
    print(f"Successfully loaded {len(all_positions)} REVE positions\n")
    print("Position range:")
    xs = [all_positions[p]['x'] for p in all_positions]
    ys = [all_positions[p]['y'] for p in all_positions]
    zs = [all_positions[p]['z'] for p in all_positions]
    print(f"  X: [{min(xs):.4f}, {max(xs):.4f}]")
    print(f"  Y: [{min(ys):.4f}, {max(ys):.4f}]")
    print(f"  Z: [{min(zs):.4f}, {max(zs):.4f}]")
    
    visualize_all_systems(all_positions)


if __name__ == "__main__":
    main()
