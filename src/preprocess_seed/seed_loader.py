"""Load and manage SEED EEG raw data (.cnt files)."""

import mne
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import struct
import os

warnings.filterwarnings('ignore')


def _read_cnt_header_info(filepath: str) -> dict:
    """
    Read Neuroscan CNT header manually to get correct sample count.
    
    The SEED dataset CNT files have corrupted n_samples fields in their headers.
    This function calculates the actual number of samples from file geometry.
    
    CNT format (Neuroscan):
    - Header is 900 bytes (SETUP struct)
    - Electrode headers: 75 bytes × n_channels (ELECTLOC structs)
    - Data starts at: 900 + (75 × n_channels)
    
    Key header fields (from http://paulbourke.net/dataformats/eeg/):
      - Offset 370-371: Number of channels (uint16, little-endian)
      - Offset 376-377: Sample rate (uint16)
      - Offset 864-867: Number of samples (uint32) - CORRUPTED in SEED files
      - Offset 886-889: Event table position (int32) - reliable for files < 2GB
      - Offset 890-893: Continuous seconds (float32)
    
    Sample count formula (from Paul Bourke):
      n_samples = (EventTablePos - (900 + 75 * n_channels)) / (2 * n_channels)
    
    Note on EventTablePos (from MNE's _compute_robust_event_table_position):
      - For files < 2GB: EventTablePos is read directly from header (reliable)
      - For files >= 2GB: EventTablePos can overflow (signed int32 max ~2.1GB)
        and must be calculated from n_samples instead
      - SEED files are ~1GB each, so EventTablePos is safe to use directly
    
    References:
      - http://paulbourke.net/dataformats/eeg/
      - https://github.com/mne-tools/mne-python/blob/main/mne/io/cnt/cnt.py
    """
    file_size = os.path.getsize(filepath)
    
    with open(filepath, 'rb') as f:
        header = f.read(900)
        
        # Number of channels (offset 370, 2 bytes, uint16 little-endian)
        n_channels = struct.unpack('<H', header[370:372])[0]
        
        # Sample rate (offset 376, 2 bytes, uint16 little-endian)
        sample_rate = struct.unpack('<H', header[376:378])[0]
        
        # Event table position (offset 886, 4 bytes)
        # MNE uses signed int32 (<i4) - for files >= 2GB this can overflow
        # SEED files are ~1GB so this is safe
        event_table_pos = struct.unpack('<i', header[886:890])[0]
    
    # Calculate data boundaries
    setup_header_size = 900
    electrode_headers_size = 75 * n_channels
    data_start = setup_header_size + electrode_headers_size
    
    # Each sample row = n_channels × 4 bytes (int32)
    # SEED data is int32 (verified via time.txt: last sample at 3,805,000 ≈ 63.4 min @ 1000Hz)
    bytes_per_sample_row = n_channels * 4
    
    # Use event table position if valid, otherwise use file size
    # MNE formula: data_size = event_offset - (data_offset + 75 * n_channels)
    if event_table_pos > data_start and event_table_pos < file_size:
        data_end = event_table_pos
    else:
        data_end = file_size
    
    data_size = data_end - data_start
    n_samples_actual = data_size // bytes_per_sample_row
    
    return {
        'n_channels': n_channels,
        'sample_rate': sample_rate,
        'n_samples': n_samples_actual,
        'header_size': setup_header_size,
        'data_start': data_start,
        'file_size': file_size
    }


class SEEDEEGLoader:
    """Load SEED dataset raw EEG .cnt files."""
    
    def __init__(self, seed_raw_dir: str, montage_path: str):
        """
        Initialize SEED EEG loader.
        
        Args:
            seed_raw_dir: Path to SEED_RAW_EEG directory
            montage_path: Path to channel_62_pos.locs file (EEGLAB format)
        """
        self.seed_raw_dir = Path(seed_raw_dir)
        self.montage_path = montage_path
        
        # Load custom montage from EEGLAB .locs file
        self.montage = mne.channels.read_custom_montage(
            montage_path, 
            head_size=1.0,
            coord_frame='head'
        )
        self.channels = self.montage.ch_names
        
        print(f"Loaded custom montage from {Path(montage_path).name} with {len(self.channels)} channels")
        
    def get_subject_sessions(self) -> dict:
        """
        Get available subject sessions.
        
        Returns:
            Dictionary mapping subject_id -> list of session indices
        """
        subjects = {}
        
        for cnt_file in sorted(self.seed_raw_dir.glob("*.cnt")):
            # Files are named like: 1_1.cnt, 1_2.cnt, etc.
            # Where first number is subject, second is session
            parts = cnt_file.stem.split('_')
            if len(parts) >= 2:
                try:
                    subject_id = int(parts[0])
                    session_id = int(parts[1])
                    
                    if subject_id not in subjects:
                        subjects[subject_id] = []
                    subjects[subject_id].append(session_id)
                except ValueError:
                    continue
        
        return subjects
    
    def load_raw(self, subject_id: int, session_id: int) -> mne.io.Raw:
        """
        Load raw EEG data for a subject-session.
        
        SEED CNT files have corrupted headers with wrong n_samples values.
        This method reads the data directly from the file, calculating the
        correct number of samples from the file size.
        
        Args:
            subject_id: Subject ID (1-15)
            session_id: Session ID (1-3)
            
        Returns:
            MNE Raw object with montage applied
        """
        cnt_file = self.seed_raw_dir / f"{subject_id}_{session_id}.cnt"
        
        if not cnt_file.exists():
            raise FileNotFoundError(f"File not found: {cnt_file}")
        
        # Read header info manually to get correct sample count
        header_info = _read_cnt_header_info(str(cnt_file))
        n_channels = header_info['n_channels']
        sfreq = header_info['sample_rate']
        n_samples = header_info['n_samples']
        setup_header_size = header_info['header_size']
        data_start = header_info['data_start']
        
        # Read channel names from CNT header (offset 900, 75 bytes per channel)
        ch_names = []
        with open(cnt_file, 'rb') as f:
            f.seek(setup_header_size)
            for i in range(n_channels):
                ch_block = f.read(75)
                # Channel name is first 10 bytes, null-terminated
                ch_name = ch_block[:10].decode('latin-1').split('\x00')[0].strip()
                ch_names.append(ch_name)
        
        # Read raw data directly from file
        # Data starts after setup header (900 bytes) + electrode headers (75 bytes × n_channels)
        with open(cnt_file, 'rb') as f:
            f.seek(data_start)
            # Read as int32 (SEED data format), reshape to (n_samples, n_channels), then transpose
            raw_data = np.fromfile(f, dtype='<i4', count=n_samples * n_channels)
            raw_data = raw_data.reshape((n_samples, n_channels)).T
        
        # Convert to float64 for MNE compatibility
        data = raw_data.astype(np.float64)
        
        # Create MNE Info object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        # Create Raw object from array
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Set montage with case insensitive matching
        # Some raw files have extra channels (M1, M2, VEO, HEO) not in montage
        raw.set_montage(self.montage, match_case=False, on_missing='ignore')
        
        # Validate montage
        self._validate_montage(raw, subject_id, session_id)
        
        return raw
    
    def _validate_montage(self, raw: mne.io.Raw, subject_id: int, session_id: int) -> None:
        """
        Validate that all EEG channels have positions.
        
        Args:
            raw: MNE Raw object with montage set
            subject_id: Subject ID (for reporting)
            session_id: Session ID (for reporting)
        """
        # Get EEG channels with position information
        # Non-EEG channels (M1, M2, VEO, HEO) are expected to have NaN positions
        chs_without_pos = []
        
        for ch_idx, ch_name in enumerate(raw.ch_names):
            ch_info = raw.info['chs'][ch_idx]
            loc = ch_info['loc'][:3]  # x, y, z coordinates
            
            # Check if position is missing (all zeros or contains NaN)
            has_position = not (np.allclose(loc, 0) or np.any(np.isnan(loc)))
            
            if not has_position and ch_name not in ['M1', 'M2', 'VEO', 'HEO']:
                # Missing position for an EEG channel
                chs_without_pos.append(ch_name)
        
        if chs_without_pos:
            print(f"Warning (Sub {subject_id}, Ses {session_id}): "
                  f"{len(chs_without_pos)} EEG channels without position: {chs_without_pos[:5]}...")
        else:
            print(f"✓ Sub {subject_id}, Ses {session_id}: All EEG channels have valid positions")
    
    def load_all_subjects_sessions(self) -> List[Tuple[int, int, mne.io.Raw]]:
        """
        Load all available subject sessions.
        
        Returns:
            List of tuples (subject_id, session_id, raw_data)
        """
        subjects = self.get_subject_sessions()
        data = []
        
        for subject_id in sorted(subjects.keys()):
            for session_id in sorted(subjects[subject_id]):
                try:
                    print(f"Loading subject {subject_id}, session {session_id}")
                    raw = self.load_raw(subject_id, session_id)
                    data.append((subject_id, session_id, raw))
                except Exception as e:
                    print(f"Error loading subject {subject_id}, session {session_id}: {e}")
                    continue
        
        return data


if __name__ == "__main__":
    seed_dir = Path(__file__).parent.parent.parent / "data" / "SEED" / "SEED_RAW_EEG"
    
    loader = SEEDEEGLoader(str(seed_dir))
    subjects = loader.get_subject_sessions()
    print(f"Found subjects: {subjects}")
    
    # Load first subject, first session as test
    raw = loader.load_raw(1, 1)
    print(f"Loaded raw data: {raw}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
