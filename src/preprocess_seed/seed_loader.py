"""Load and manage SEED EEG raw data (.cnt files)."""

import mne
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


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
        
        Args:
            subject_id: Subject ID (1-15)
            session_id: Session ID (1-3)
            
        Returns:
            MNE Raw object with montage applied
        """
        cnt_file = self.seed_raw_dir / f"{subject_id}_{session_id}.cnt"
        
        if not cnt_file.exists():
            raise FileNotFoundError(f"File not found: {cnt_file}")
        
        # Load with MNE - Neuroscan CNT format
        # Don't preload initially due to potential header issues
        # Some files have corrupted sample counts in the header
        try:
            raw = mne.io.read_raw_cnt(cnt_file, preload=False, verbose=False, data_format='int16')
        except Exception as e:
            raise RuntimeError(f"Could not read {cnt_file}: {e}")
        
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
                # Missing position for an EEG channel - this is a real issue
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
