"""
Deep inspection of Neuroscan CNT file format.

The Neuroscan CNT format is a binary format with this structure:

╔══════════════════════════════════════════════════════════════════════════════╗
║                           NEUROSCAN CNT FORMAT                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ SETUP HEADER (900 bytes)                                                │ ║
║  │   - General recording info                                              │ ║
║  │   - Offset 370-371: Number of channels (uint16)                         │ ║
║  │   - Offset 376-377: Sample rate (uint16)                                │ ║
║  │   - Offset 864-867: Number of samples (uint32) ← OFTEN CORRUPTED!       │ ║
║  │   - Offset 886-889: Event table offset (uint32)                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ ELECTRODE HEADERS (75 bytes × n_channels)                               │ ║
║  │   For each channel:                                                     │ ║
║  │   - Offset 0-9: Channel label (10 bytes, null-terminated string)        │ ║
║  │   - Offset 59-62: Sensitivity (float32)                                 │ ║
║  │   - Offset 63-66: Calibration (float32)                                 │ ║
║  │   - Other fields: baseline, physical min/max, etc.                      │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ EEG DATA (variable size)                                                │ ║
║  │   - Multiplexed int16 samples                                           │ ║
║  │   - Layout: [ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ch1_s1, ...]          │ ║
║  │   - Each sample = 2 bytes × n_channels                                  │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ EVENT TABLE (at offset from header, optional)                           │ ║
║  │   - Markers, triggers, annotations                                      │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Memory calculation:
  - Data starts at: 900 + (75 × n_channels) bytes
  - Data ends at: either event_table_offset OR end of file
  - Data size = end_offset - start_offset
  - n_samples = data_size / (n_channels × 2)  [each sample is int16 = 2 bytes]
"""

import struct
import os
from pathlib import Path
import numpy as np


def inspect_cnt_file_detailed(filepath: str) -> dict:
    """
    Detailed inspection of a CNT file showing all header fields.
    """
    file_size = os.path.getsize(filepath)
    
    with open(filepath, 'rb') as f:
        # Read entire setup header (900 bytes)
        setup_header = f.read(900)
        
        # ===== SETUP HEADER FIELDS =====
        # These offsets are from the Neuroscan CNT specification
        
        # Revision string (offset 0, 12 bytes)
        rev = setup_header[0:12].decode('latin-1', errors='replace').strip('\x00')
        
        # Number of channels (offset 370, uint16)
        n_channels = struct.unpack('<H', setup_header[370:372])[0]
        
        # Sample rate (offset 376, uint16) 
        sample_rate = struct.unpack('<H', setup_header[376:378])[0]
        
        # Number of samples - THIS IS THE CORRUPTED FIELD
        # Offset 864 (4 bytes, uint32)
        n_samples_header = struct.unpack('<I', setup_header[864:868])[0]
        
        # Event table offset (offset 886, uint32)
        event_table_offset = struct.unpack('<I', setup_header[886:890])[0]
        
        # ===== ELECTRODE HEADERS =====
        electrode_header_size = 75
        electrode_headers_start = 900
        data_start = electrode_headers_start + (electrode_header_size * n_channels)
        
        # Read channel info
        channels = []
        for i in range(n_channels):
            f.seek(electrode_headers_start + i * electrode_header_size)
            ch_header = f.read(electrode_header_size)
            
            # Channel name (first 10 bytes, null-terminated)
            ch_name = ch_header[0:10].decode('latin-1', errors='replace').split('\x00')[0].strip()
            
            # Sensitivity and calibration (used for scaling)
            # Offset 59-62: sensitivity (float32)
            # Offset 63-66: calibration (float32)
            sensitivity = struct.unpack('<f', ch_header[59:63])[0]
            calibration = struct.unpack('<f', ch_header[63:67])[0]
            
            channels.append({
                'name': ch_name,
                'sensitivity': sensitivity,
                'calibration': calibration
            })
        
        # ===== DATA SIZE CALCULATION =====
        # The key insight: we can calculate actual data size from file structure
        
        # Method 1: Use event table offset (if valid)
        # Method 2: Use file size
        
        if event_table_offset > data_start and event_table_offset < file_size:
            data_end = event_table_offset
            data_end_source = "event_table_offset"
        else:
            data_end = file_size
            data_end_source = "file_size"
        
        data_size = data_end - data_start
        bytes_per_sample = n_channels * 2  # int16 = 2 bytes
        n_samples_calculated = data_size // bytes_per_sample
        
        # ===== VERIFY BY READING SOME DATA =====
        f.seek(data_start)
        # Read first few samples to verify format
        first_samples = np.frombuffer(f.read(bytes_per_sample * 10), dtype='<i2')
        first_samples = first_samples.reshape((10, n_channels))
        
        # Read last few samples
        f.seek(data_end - bytes_per_sample * 10)
        last_samples = np.frombuffer(f.read(bytes_per_sample * 10), dtype='<i2')
        last_samples = last_samples.reshape((10, n_channels))
    
    return {
        'filepath': filepath,
        'file_size': file_size,
        'revision': rev,
        'setup_header_size': 900,
        'n_channels': n_channels,
        'sample_rate': sample_rate,
        'n_samples_from_header': n_samples_header,
        'event_table_offset': event_table_offset,
        'electrode_header_size': electrode_header_size,
        'data_start_offset': data_start,
        'data_end_offset': data_end,
        'data_end_source': data_end_source,
        'data_size_bytes': data_size,
        'bytes_per_sample_row': bytes_per_sample,
        'n_samples_calculated': n_samples_calculated,
        'duration_seconds': n_samples_calculated / sample_rate,
        'duration_minutes': n_samples_calculated / sample_rate / 60,
        'channels': channels,
        'first_10_samples': first_samples,
        'last_10_samples': last_samples,
        # Memory estimates
        'memory_if_using_header_gb': (n_samples_header * n_channels * 8) / (1024**3),
        'memory_actual_gb': (n_samples_calculated * n_channels * 8) / (1024**3),
    }


def compare_two_files(file1: str, file2: str):
    """Compare structure of two CNT files to verify format consistency."""
    info1 = inspect_cnt_file_detailed(file1)
    info2 = inspect_cnt_file_detailed(file2)
    
    print("=" * 80)
    print("COMPARING TWO CNT FILES")
    print("=" * 80)
    
    print(f"\nFile 1: {Path(file1).name}")
    print(f"File 2: {Path(file2).name}")
    
    # Compare key structural elements
    fields_to_compare = [
        ('setup_header_size', 'Setup header size'),
        ('n_channels', 'Number of channels'),
        ('sample_rate', 'Sample rate'),
        ('electrode_header_size', 'Electrode header size'),
        ('bytes_per_sample_row', 'Bytes per sample row'),
    ]
    
    print(f"\n{'Field':<30} {'File 1':<20} {'File 2':<20} {'Match'}")
    print("-" * 80)
    
    for field, label in fields_to_compare:
        v1 = info1[field]
        v2 = info2[field]
        match = "✓" if v1 == v2 else "✗ DIFFERENT"
        print(f"{label:<30} {str(v1):<20} {str(v2):<20} {match}")
    
    # Compare variable fields
    print(f"\n{'Variable Field':<30} {'File 1':<20} {'File 2':<20}")
    print("-" * 80)
    
    variable_fields = [
        ('file_size', 'File size (bytes)'),
        ('n_samples_from_header', 'n_samples (header) - CORRUPTED'),
        ('n_samples_calculated', 'n_samples (calculated)'),
        ('duration_minutes', 'Duration (minutes)'),
        ('data_start_offset', 'Data start offset'),
        ('data_end_source', 'Data end determined by'),
    ]
    
    for field, label in variable_fields:
        v1 = info1[field]
        v2 = info2[field]
        if isinstance(v1, float):
            print(f"{label:<30} {v1:<20.2f} {v2:<20.2f}")
        else:
            print(f"{label:<30} {str(v1):<20} {str(v2):<20}")
    
    # Compare channel names
    print(f"\nChannel names match: ", end="")
    ch1 = [c['name'] for c in info1['channels']]
    ch2 = [c['name'] for c in info2['channels']]
    if ch1 == ch2:
        print(f"✓ All {len(ch1)} channels identical")
    else:
        print(f"✗ Different!")
        print(f"  File 1: {ch1[:5]}...")
        print(f"  File 2: {ch2[:5]}...")
    
    return info1, info2


def print_detailed_inspection(filepath: str):
    """Print detailed inspection of a single file."""
    info = inspect_cnt_file_detailed(filepath)
    
    print("=" * 80)
    print(f"DETAILED CNT FILE INSPECTION: {Path(filepath).name}")
    print("=" * 80)
    
    print(f"\n--- FILE STRUCTURE ---")
    print(f"File size: {info['file_size']:,} bytes ({info['file_size']/1024/1024:.2f} MB)")
    print(f"Revision string: '{info['revision']}'")
    
    print(f"\n--- SETUP HEADER (900 bytes) ---")
    print(f"Number of channels: {info['n_channels']}")
    print(f"Sample rate: {info['sample_rate']} Hz")
    print(f"n_samples from header: {info['n_samples_from_header']:,} ← THIS IS CORRUPTED!")
    print(f"Event table offset: {info['event_table_offset']:,}")
    
    print(f"\n--- ELECTRODE HEADERS ({info['electrode_header_size']} × {info['n_channels']} = {info['electrode_header_size'] * info['n_channels']} bytes) ---")
    print(f"Channels: {[c['name'] for c in info['channels'][:10]]}... ({len(info['channels'])} total)")
    
    print(f"\n--- DATA SECTION ---")
    print(f"Data starts at byte: {info['data_start_offset']:,}")
    print(f"Data ends at byte: {info['data_end_offset']:,} (from {info['data_end_source']})")
    print(f"Data size: {info['data_size_bytes']:,} bytes ({info['data_size_bytes']/1024/1024:.2f} MB)")
    print(f"Bytes per sample row: {info['bytes_per_sample_row']} (= {info['n_channels']} channels × 2 bytes/int16)")
    
    print(f"\n--- CALCULATED VALUES ---")
    print(f"Actual n_samples: {info['n_samples_calculated']:,}")
    print(f"Duration: {info['duration_seconds']:.2f} seconds ({info['duration_minutes']:.2f} minutes)")
    
    print(f"\n--- MEMORY COMPARISON ---")
    print(f"If using header n_samples: {info['memory_if_using_header_gb']:.2f} GB ← WOULD CRASH!")
    print(f"Actual memory needed: {info['memory_actual_gb']:.3f} GB")
    
    print(f"\n--- DATA VERIFICATION ---")
    print(f"First sample (all channels): {info['first_10_samples'][0]}")
    print(f"Last sample (all channels): {info['last_10_samples'][-1]}")
    print(f"Data range in first 10 samples: [{info['first_10_samples'].min()}, {info['first_10_samples'].max()}]")
    print(f"Data range in last 10 samples: [{info['last_10_samples'].min()}, {info['last_10_samples'].max()}]")
    
    return info


if __name__ == "__main__":
    seed_dir = Path(__file__).parent.parent.parent / "data" / "SEED" / "SEED_RAW_EEG"
    
    # Detailed inspection of working file
    print("\n" + "=" * 80)
    print("INSPECTING SUBJECT 1 SESSION 1 (was working)")
    print("=" * 80)
    info1 = print_detailed_inspection(str(seed_dir / "1_1.cnt"))
    
    # Detailed inspection of previously failing file
    print("\n" + "=" * 80)
    print("INSPECTING SUBJECT 2 SESSION 1 (was failing)")
    print("=" * 80)
    info2 = print_detailed_inspection(str(seed_dir / "2_1.cnt"))
    
    # Compare them
    print("\n")
    compare_two_files(str(seed_dir / "1_1.cnt"), str(seed_dir / "2_1.cnt"))
