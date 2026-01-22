"""
Pytest configuration and fixtures for data loader tests.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import get_data_loader


@pytest.fixture(scope="session")
def zl_dataset():
    """
    Load ZL_Dataset once per test session.
    This fixture is shared across all tests.
    """
    loader = get_data_loader('zl')
    return loader


@pytest.fixture
def sample_subject_session(zl_dataset):
    """
    Get the first available subject/session pair.
    """
    pairs = zl_dataset.get_all_subject_sessions()
    if not pairs:
        pytest.skip("No subject/session pairs found in dataset")
    return pairs[0]
