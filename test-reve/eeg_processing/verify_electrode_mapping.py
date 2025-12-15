"""
Verify that CapTrak electrode numbers match XDF channel order.
This ensures electrode 1 is at the right position, electrode 2 at the right position, etc.
"""

import json
from pathlib import Path
import pyxdf
from config import DATA_ROOT


def verify_electrode_mapping():
    """Check if CapTrak electrodes map correctly to XDF channels."""
    
    # Load electrode positions
    json_path = Path(__file__).parent / "electrode_positions.json"
    if not json_path.exists():
        print("electrode_positions.json not found")
        return
    
    with open(json_path, 'r') as f:
        electrodes = json.load(f)
    
    electrode_names = sorted([int(k) for k in electrodes.keys()])
    print(f"Electrodes from CapTrak: {len(electrode_names)}")
    print(f"First 10: {electrode_names[:10]}")
    print(f"Last 10: {electrode_names[-10:]}\n")
    
    # Load XDF and get channel labels
    data_path = Path(DATA_ROOT)
    xdf_files = list(data_path.glob("**/eeg/*.xdf"))
    
    if not xdf_files:
        print("No XDF files found")
        return
    
    xdf_file = xdf_files[0]
    streams, _ = pyxdf.load_xdf(str(xdf_file))
    
    for stream in streams:
        if stream['info']['type'][0] == 'EEG':
            if 'desc' in stream['info'] and stream['info']['desc']:
                desc = stream['info']['desc'][0]
                if 'channels' in desc:
                    # channels is a list with one dict that contains 'channel' key
                    channels_list = desc['channels'][0]['channel']
                    
                    print(f"Channels in XDF EEG stream: {len(channels_list)}\n")
                    
                    # Extract all channel labels
                    xdf_labels = []
                    for ch_dict in channels_list:
                        label = ch_dict.get('label', ['Unknown'])[0] if isinstance(ch_dict.get('label'), list) else ch_dict.get('label', 'Unknown')
                        xdf_labels.append(label)
                    
                    print(f"XDF Channel labels:")
                    print(f"First 10: {xdf_labels[:10]}")
                    print(f"Last 10: {xdf_labels[-10:]}\n")
                    
                    # Filter to EEG only (exclude AUX and Markers)
                    eeg_labels = [label for label in xdf_labels if label not in ['AUX_1', 'AUX_2', 'Markers']]
                    print(f"EEG-only labels: {len(eeg_labels)}")
                    print(f"First 10: {eeg_labels[:10]}")
                    print(f"Last 10: {eeg_labels[-10:]}\n")
                    
                    # Try to convert to integers to check if they match electrode numbers
                    try:
                        eeg_nums = [int(label) for label in eeg_labels]
                        print(f"EEG labels as numbers:")
                        print(f"First 10: {eeg_nums[:10]}")
                        print(f"Last 10: {eeg_nums[-10:]}\n")
                        
                        # Check if they match
                        if eeg_nums == electrode_names:
                            print("✓ PERFECT MATCH!")
                            print("CapTrak electrodes 1-96 correspond exactly to XDF EEG channels 1-96")
                            print("Electrode N is at the same position as Channel N")
                        else:
                            print("✗ MISMATCH!")
                            print(f"CapTrak electrodes: {electrode_names}")
                            print(f"XDF EEG channels: {eeg_nums}")
                            
                            # Find differences
                            for i, (cap, xdf) in enumerate(zip(electrode_names, eeg_nums)):
                                if cap != xdf:
                                    print(f"  Position {i}: CapTrak={cap}, XDF={xdf}")
                    except ValueError:
                        print("✗ XDF labels are not numeric, cannot verify mapping")
                        print("Labels appear to be:", eeg_labels[:5])


if __name__ == "__main__":
    verify_electrode_mapping()
