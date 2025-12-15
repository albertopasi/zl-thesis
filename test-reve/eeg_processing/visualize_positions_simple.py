"""
Simple electrode position visualization using MNE.
Uses actual channel labels from XDF file.
"""

import json
from pathlib import Path
import numpy as np
import mne
import pyxdf
from config import DATA_ROOT


def load_positions():
    """Load positions from JSON file."""
    json_path = Path(__file__).parent / "electrode_positions.json"
    
    if json_path.exists():
        print(f"✓ Loading positions from: {json_path.name}\n")
        with open(json_path, 'r') as f:
            electrodes_data = json.load(f)
        # Convert string keys back to int
        return {int(k): v for k, v in electrodes_data.items()}
    
    print("JSON file not found")
    return None


def get_eeg_channel_labels():
    """Extract actual channel labels from XDF file."""
    data_path = Path(DATA_ROOT)
    xdf_files = list(data_path.glob("**/eeg/*.xdf"))
    
    if not xdf_files:
        return None
    
    xdf_file = xdf_files[0]
    streams, _ = pyxdf.load_xdf(str(xdf_file))
    
    for stream in streams:
        if stream['info']['type'][0] == 'EEG':
            if 'desc' in stream['info'] and stream['info']['desc']:
                desc = stream['info']['desc'][0]
                if 'channels' in desc:
                    # channels is a list with one dict that contains 'channel' key
                    channels_list = desc['channels'][0]['channel']
                    
                    # Extract all channel labels
                    labels = []
                    for ch_dict in channels_list:
                        label = ch_dict.get('label', ['Unknown'])[0] if isinstance(ch_dict.get('label'), list) else ch_dict.get('label', 'Unknown')
                        labels.append(label)
                    
                    print(f"Extracted {len(labels)} channel labels from XDF")
                    return labels
    
    return None


def plot_electrode_positions(electrodes):
    """Visualize electrode positions using MNE."""
    
    if not electrodes:
        print("No electrodes to plot")
        return
    
    print(f"Plotting {len(electrodes)} electrode positions...\n")
    
    # Get actual channel labels from XDF
    ch_labels = get_eeg_channel_labels()
    if not ch_labels:
        print("Could not load channel labels from XDF")
        return
    
    # Filter to only EEG channels (exclude AUX and Markers)
    eeg_labels = [label for label in ch_labels if label not in ['AUX_1', 'AUX_2', 'Markers']]
    
    print(f"Using {len(eeg_labels)} EEG channel labels from XDF")
    print(f"First 10 labels: {eeg_labels[:10]}\n")
    
    # Extract positions in order matching electrode numbers
    positions = {}
    ch_names_list = []
    
    for name in sorted([int(k) for k in electrodes.keys()]):
        el = electrodes[name]
        # Convert mm to meters (MNE standard)
        x, y, z = el['x'] / 1000, el['y'] / 1000, el['z'] / 1000
        
        # Use actual channel label from XDF (electrode number maps to channel label)
        if name - 1 < len(eeg_labels):
            ch_name = str(eeg_labels[name - 1])  # electrode 1 -> index 0
            print(f"Electrode {name} -> Channel label: {ch_name}")
        else:
            ch_name = f"E{name}"
            print(f"Electrode {name} -> No label found, using: {ch_name}")
        
        positions[ch_name] = np.array([x, y, z])
        ch_names_list.append(ch_name)
    
    print(f"\nCreating visualization with {len(ch_names_list)} channels...\n")
    
    # Create info
    info = mne.create_info(ch_names_list, sfreq=500, ch_types='eeg')
    raw = mne.io.RawArray(np.zeros((len(ch_names_list), 1000)), info)
    
    # Create montage from positions
    montage = mne.channels.make_dig_montage(
        ch_pos=positions,
        coord_frame='head'
    )
    
    raw.set_montage(montage)
    
    # Plot in 3D with labels
    print("Displaying 3D view...")
    fig = mne.viz.plot_sensors(raw.info, kind='3d', title='3D Electrode Positions', show_names=True)
    
    # Plot in 2D with labels
    print("Displaying 2D view...")
    fig = mne.viz.plot_sensors(raw.info, kind='topomap', title='2D Electrode Layout', show_names=True)
    
    # Keep plots open
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    electrodes = load_positions()
    
    if electrodes:
        plot_electrode_positions(electrodes)
    else:
        print("Could not load electrode positions")
