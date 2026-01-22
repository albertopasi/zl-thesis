"""
Dataset registry and factory for managing multiple data sources.
"""

from typing import Dict, Type, Optional
from pathlib import Path

from .base import DataLoader
from .zl_dataset import ZLDataset


class DatasetRegistry:
    """Registry for different dataset loaders."""
    
    _loaders: Dict[str, Type[DataLoader]] = {
        'zl': ZLDataset,
    }
    
    @classmethod
    def register(cls, name: str, loader_class: Type[DataLoader]) -> None:
        """
        Register a new dataset loader.
        
        Args:
            name: Name to identify this dataset
            loader_class: DataLoader subclass
        """
        if not issubclass(loader_class, DataLoader):
            raise TypeError(f"{loader_class} must be a subclass of DataLoader")
        cls._loaders[name] = loader_class
        print(f"Registered dataset loader: {name}")
    
    @classmethod
    def get_loader(cls, name: str, dataset_root: str, **kwargs) -> DataLoader:
        """
        Get a data loader instance.
        
        Args:
            name: Dataset name ('current', 'custom', etc.)
            dataset_root: Root directory of the dataset
            **kwargs: Additional arguments for the loader
            
        Returns:
            DataLoader instance
            
        Raises:
            ValueError: If dataset name not found
        """
        if name not in cls._loaders:
            available = ', '.join(cls._loaders.keys())
            raise ValueError(
                f"Unknown dataset: '{name}'. Available: {available}"
            )
        
        loader_class = cls._loaders[name]
        return loader_class(dataset_root, **kwargs)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered dataset loaders."""
        return list(cls._loaders.keys())


def get_data_loader(dataset_name: str = 'zl', 
                    dataset_root: Optional[str] = None,
                    **kwargs) -> DataLoader:
    """
    Convenience function to get a data loader.
    
    Args:
        dataset_name: Name of dataset ('zl', 'custom', etc.)
        dataset_root: Root directory of dataset. If None, tries to auto-detect.
        **kwargs: Additional loader arguments
        
    Returns:
        DataLoader instance
    """
    if dataset_root is None:
        # Auto-detect: look for data folder in parent directories
        dataset_root = _auto_detect_dataset_root()
        if dataset_root is None:
            raise FileNotFoundError(
                "Could not auto-detect dataset root. Please provide dataset_root explicitly."
            )
    
    return DatasetRegistry.get_loader(dataset_name, dataset_root, **kwargs)


def _auto_detect_dataset_root() -> Optional[str]:
    """
    Auto-detect dataset root by searching for common patterns.
    Handles nested structures like data/Zander Labs/sub-*.
    
    Returns:
        Path to dataset root or None
    """
    # Search upward from current directory
    current = Path.cwd()
    
    for _ in range(5):  # Search up to 5 levels
        candidate = current / 'data'
        if candidate.exists() and candidate.is_dir():
            # Check if it directly has sub-* directories
            if any(d.name.startswith('sub-') for d in candidate.iterdir() if d.is_dir()):
                return str(candidate)
            
            # Check if any subdirectory has sub-* directories (nested structure)
            for item in candidate.iterdir():
                if item.is_dir() and any(d.name.startswith('sub-') for d in item.iterdir() if d.is_dir()):
                    return str(item)
        
        current = current.parent
    
    return None
