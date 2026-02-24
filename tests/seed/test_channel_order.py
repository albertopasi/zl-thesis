"""Test channel order verification against channel-order.xlsx."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from preprocess_seed.seed_loader import SEEDEEGLoader


class TestChannelOrder:
    """Test that loaded channels match expected order from Excel file."""
    
    @pytest.fixture
    def config_paths(self):
        """Get configuration paths."""
        data_dir = Path(__file__).parent.parent.parent / "data" / "SEED"
        return {
            'seed_raw_dir': data_dir / "SEED_RAW_EEG",
            'montage_file': data_dir / "channel_62_pos.locs",
            'channel_order_file': data_dir / "channel-order.xlsx"
        }
    
    @pytest.fixture
    def loader(self, config_paths):
        """Create loader instance."""
        return SEEDEEGLoader(
            str(config_paths['seed_raw_dir']),
            str(config_paths['montage_file'])
        )
    
    @pytest.fixture
    def expected_channel_order(self, config_paths):
        """Load expected channel order from Excel."""
        df = pd.read_excel(config_paths['channel_order_file'], header=None)
        print(f"\nExcel columns: {df.columns.tolist()}")
        print(f"Excel shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        return df
    
    @pytest.fixture
    def raw_data(self, loader):
        """Load raw data for testing."""
        return loader.load_raw(1, 1)
    
    def test_channel_order_matches_excel(self, raw_data, expected_channel_order):
        """Test that raw data EEG channel order matches Excel file order."""
        # Get channels from raw data
        raw_channels = raw_data.ch_names
        
        print(f"\nRaw data total channels ({len(raw_channels)}): {raw_channels}")
        
        # Extract channel names from dataframe (column 0, all rows)
        expected_channels = expected_channel_order[0].tolist()
        expected_channels_lower = [ch.lower().strip() if isinstance(ch, str) else str(ch) for ch in expected_channels]
        
        print(f"\nExpected channels from Excel ({len(expected_channels)}): {expected_channels}")
        
        # Filter raw channels to only include expected EEG channels
        raw_channels_lower = [ch.lower().strip() for ch in raw_channels]
        eeg_channels_filtered = []
        eeg_indices = []
        
        for i, raw_ch in enumerate(raw_channels_lower):
            if raw_ch in expected_channels_lower:
                eeg_channels_filtered.append(raw_channels[i])
                eeg_indices.append(i)
        
        print(f"\nFiltered EEG channels from raw ({len(eeg_channels_filtered)}): {eeg_channels_filtered}")
        
        # Now compare the order of EEG channels
        eeg_channels_normalized = [ch.lower().strip() for ch in eeg_channels_filtered]
        
        assert len(eeg_channels_filtered) == len(expected_channels), \
            f"EEG channel count mismatch: raw has {len(eeg_channels_filtered)}, expected has {len(expected_channels)}"
        
        mismatches = []
        for i, (raw, expected) in enumerate(zip(eeg_channels_normalized, expected_channels_lower)):
            if raw != expected:
                mismatches.append((i, raw, expected))
        
        if mismatches:
            print(f"\n❌ Found {len(mismatches)} EEG channel order mismatches:")
            for idx, raw, expected in mismatches[:10]:  # Show first 10
                print(f"  Position {idx}: got '{raw}', expected '{expected}'")
            pytest.fail(f"EEG channel order mismatch at {len(mismatches)} positions")
        else:
            print(f"\n✓ All {len(eeg_channels_filtered)} EEG channels match expected order")
    
    def test_all_expected_channels_present(self, raw_data, expected_channel_order):
        """Test that all expected channels are present in raw data."""
        raw_channels_lower = [ch.lower().strip() for ch in raw_data.ch_names]
        
        # Extract channel names from dataframe (column 0, all rows)
        expected_channels = expected_channel_order[0].tolist()
        expected_channels_lower = [ch.lower().strip() if isinstance(ch, str) else str(ch) for ch in expected_channels]
        
        missing = []
        for expected in expected_channels_lower:
            if expected not in raw_channels_lower:
                missing.append(expected)
        
        if missing:
            print(f"\n❌ Missing channels: {missing}")
            pytest.fail(f"Missing {len(missing)} expected channels")
        else:
            print(f"\n✓ All {len(expected_channels_lower)} expected channels present")
    
    def test_no_extra_channels(self, raw_data, expected_channel_order):
        """Test and report any extra non-EEG channels in raw data."""
        # Extract channel names from dataframe (column 0, all rows)
        expected_channels = expected_channel_order[0].tolist()
        expected_channels_lower = [ch.lower().strip() if isinstance(ch, str) else str(ch) for ch in expected_channels]
        raw_channels_lower = [ch.lower().strip() for ch in raw_data.ch_names]
        
        extra = []
        for raw in raw_channels_lower:
            if raw not in expected_channels_lower:
                extra.append(raw)
        
        if extra:
            print(f"\n⚠ Extra non-EEG channels in raw data ({len(extra)}): {extra}")
            print("These are likely reference (M1, M2) and EOG (VEO, HEO) channels")
            print("✓ This is expected and correct for SEED data preprocessing")
        else:
            print(f"\n✓ No extra channels (all {len(raw_channels_lower)} are expected)")


def test_examine_excel_structure():
    """Examine and print the structure of channel-order.xlsx."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "SEED"
    excel_file = data_dir / "channel-order.xlsx"
    
    if not excel_file.exists():
        pytest.skip(f"Channel order file not found: {excel_file}")
    
    df = pd.read_excel(excel_file, header=None)
    
    print(f"\n{'='*80}")
    print("CHANNEL ORDER EXCEL FILE STRUCTURE")
    print(f"{'='*80}")
    print(f"File: {excel_file}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nAll channel names:")
    for i, ch in enumerate(df[0].tolist()):
        print(f"  {i+1:2d}. {ch}")
    print(f"\nData types:")
    print(df.dtypes)
