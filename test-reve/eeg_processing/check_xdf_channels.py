"""
Check actual channel labels from XDF file.
"""

from pathlib import Path
from config import DATA_ROOT
import pyxdf


data_path = Path(DATA_ROOT)
xdf_files = list(data_path.glob("**/eeg/*.xdf"))

if xdf_files:
    xdf_file = xdf_files[0]
    print(f"Loading: {xdf_file.name}\n")
    
    streams, header = pyxdf.load_xdf(str(xdf_file))
    
    for stream_idx, stream in enumerate(streams):
        stream_name = stream['info']['name'][0]
        stream_type = stream['info']['type'][0]
        
        if stream_type == 'EEG':
            print(f"EEG Stream: {stream_name}")
            print(f"Type: {stream_type}")
            print(f"Channels: {int(stream['info']['channel_count'][0])}\n")
            
            # Get channel descriptions
            if 'desc' in stream['info'] and stream['info']['desc']:
                desc = stream['info']['desc'][0]
                if 'channels' in desc:
                    channels = desc['channels']
                    print("Channel Labels from XDF:")
                    for i, ch in enumerate(channels):
                        # Extract label properly
                        ch_dict = ch.get('channel', [{}])[0] if isinstance(ch.get('channel'), list) else ch
                        label = ch_dict.get('label', ['Unknown'])[0] if isinstance(ch_dict.get('label'), list) else ch_dict.get('label', f'CH{i}')
                        print(f"  {i}: {label}")
else:
    print(f"No XDF files found in {DATA_ROOT}")
