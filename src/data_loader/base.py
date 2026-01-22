"""
Abstract base class for data loaders.
Provides common interface for all dataset implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class DataLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, dataset_root: str):
        """
        Initialize data loader.
        
        Args:
            dataset_root: Root directory of the dataset
        """
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        
        self.subjects = []
        self.sessions = {}  # {subject_id: [session_ids]}
        self._discover_subjects()
    
    @abstractmethod
    def _discover_subjects(self):
        """
        Discover all subjects in the dataset.
        Must populate self.subjects and self.sessions.
        """
        pass
    
    @abstractmethod
    def get_eeg_file_path(self, subject_id: str, session_id: str) -> Path:
        """
        Get path to EEG data file for a subject/session.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            
        Returns:
            Path to EEG file
        """
        pass
    
    @abstractmethod
    def load_subject_data(self, subject_id: str, session_id: str) -> Dict[str, Any]:
        """
        Load raw data for a subject/session.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            
        Returns:
            dict with keys: 'eeg', 'markers', 'eeg_timestamps', 'marker_timestamps', 'sampling_rate', 'channel_labels'
        """
        pass
    
    def get_all_subjects(self) -> List[str]:
        """Get list of all available subjects."""
        return sorted(self.subjects)
    
    def get_sessions_for_subject(self, subject_id: str) -> List[str]:
        """Get sessions available for a subject."""
        return self.sessions.get(subject_id, [])
    
    def get_all_subject_sessions(self) -> List[Tuple[str, str]]:
        """
        Get all (subject, session) pairs.
        
        Returns:
            List of (subject_id, session_id) tuples
        """
        pairs = []
        for subject_id in sorted(self.subjects):
            for session_id in sorted(self.sessions.get(subject_id, [])):
                pairs.append((subject_id, session_id))
        return pairs
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(subjects={len(self.subjects)}, root={self.dataset_root})"
