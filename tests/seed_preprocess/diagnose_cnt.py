"""Diagnose CNT file headers to find corruption issues."""

import struct
from pathlib import Path
import os


def read_cnt_header(filepath: str) -> dict:
    """
    Read Neuroscan CNT header manually to diagnose issues.
    
    CNT format (Neuroscan):
    - Header is 900 bytes
    - Key fields:
      - Offset 370-372: Number of channels (2 bytes, little-endian)
      - Offset 864-868: Number of samples (4 bytes, little-endian) - OFTEN CORRUPTED
      - Offset 376-378: Sample rate (2 bytes)
    """
    info = {'filepath': filepath}
    
    file_size = os.path.getsize(filepath)
    info['file_size_bytes'] = file_size
    info['file_size_mb'] = file_size / (1024 * 1024)
    
    with open(filepath, 'rb') as f:
        # Read full header
        header = f.read(900)
        
        # Number of channels (offset 370, 2 bytes)
        n_channels = struct.unpack('<H', header[370:372])[0]
        info['n_channels_header'] = n_channels
        
        # Sample rate (offset 376, 2 bytes) 
        sample_rate = struct.unpack('<H', header[376:378])[0]
        info['sample_rate_header'] = sample_rate
        
        # Number of samples from header (offset 864, 4 bytes) - often corrupted!
        n_samples_header = struct.unpack('<I', header[864:868])[0]
        info['n_samples_header'] = n_samples_header
        
        # Also check offset 886 for event table position (can help estimate actual data)
        event_table_pos = struct.unpack('<I', header[886:890])[0]
        info['event_table_pos'] = event_table_pos
        
    # Calculate expected samples from file size
    # Data starts at offset 900, each sample is n_channels * 2 bytes (int16)
    header_size = 900
    bytes_per_sample = n_channels * 2  # int16 = 2 bytes per channel
    
    # Use event table position if valid, otherwise use file size
    if event_table_pos > header_size and event_table_pos < file_size:
        data_size = event_table_pos - header_size
        info['data_end_marker'] = 'event_table'
    else:
        data_size = file_size - header_size
        info['data_end_marker'] = 'file_end'
    
    n_samples_calculated = data_size // bytes_per_sample
    info['n_samples_calculated'] = n_samples_calculated
    
    # Calculate duration
    if sample_rate > 0:
        info['duration_header_sec'] = n_samples_header / sample_rate
        info['duration_header_min'] = info['duration_header_sec'] / 60
        info['duration_calculated_sec'] = n_samples_calculated / sample_rate
        info['duration_calculated_min'] = info['duration_calculated_sec'] / 60
    
    # Memory that would be allocated
    info['memory_header_gb'] = (n_samples_header * n_channels * 8) / (1024**3)  # float64
    info['memory_calculated_gb'] = (n_samples_calculated * n_channels * 8) / (1024**3)
    
    # Is header likely corrupted?
    info['header_corrupted'] = n_samples_header > n_samples_calculated * 1.1
    
    return info


def diagnose_all_cnt_files(seed_raw_dir: str):
    """Diagnose all CNT files in directory."""
    seed_path = Path(seed_raw_dir)
    
    print("=" * 80)
    print("CNT FILE HEADER DIAGNOSIS")
    print("=" * 80)
    
    cnt_files = sorted(seed_path.glob("*.cnt"))
    print(f"\nFound {len(cnt_files)} CNT files\n")
    
    corrupted = []
    valid = []
    
    for cnt_file in cnt_files:
        try:
            info = read_cnt_header(str(cnt_file))
            
            status = "✗ CORRUPTED" if info['header_corrupted'] else "✓ OK"
            
            print(f"{cnt_file.name}:")
            print(f"  File size: {info['file_size_mb']:.1f} MB")
            print(f"  Channels: {info['n_channels_header']}, Sample rate: {info['sample_rate_header']} Hz")
            print(f"  Header n_samples: {info['n_samples_header']:,} ({info.get('duration_header_min', 0):.1f} min)")
            print(f"  Calculated n_samples: {info['n_samples_calculated']:,} ({info.get('duration_calculated_min', 0):.1f} min)")
            print(f"  Memory (header): {info['memory_header_gb']:.2f} GB vs calculated: {info['memory_calculated_gb']:.3f} GB")
            print(f"  Status: {status}")
            print()
            
            if info['header_corrupted']:
                corrupted.append((cnt_file.name, info))
            else:
                valid.append((cnt_file.name, info))
                
        except Exception as e:
            print(f"{cnt_file.name}: ERROR - {e}")
            print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Valid files: {len(valid)}")
    print(f"Corrupted files: {len(corrupted)}")
    
    if corrupted:
        print(f"\nCorrupted files that need header fix:")
        for name, info in corrupted:
            print(f"  - {name}: header says {info['n_samples_header']:,} samples, "
                  f"actual ~{info['n_samples_calculated']:,}")
    
    return valid, corrupted


if __name__ == "__main__":
    from pathlib import Path
    
    seed_dir = Path(__file__).parent.parent.parent / "data" / "SEED" / "SEED_RAW_EEG"
    diagnose_all_cnt_files(str(seed_dir))
