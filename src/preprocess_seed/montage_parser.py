"""Parse EEGLAB .locs files and create MNE montages."""

import numpy as np
from pathlib import Path
from mne import create_info
from mne.channels import make_dig_montage


def parse_locs_file(locs_file_path: str) -> dict:
    """
    Parse EEGLAB .locs file format.
    
    Format: index  x_angle  y_angle  channel_name
    (angles in degrees, normalized to sphere)
    
    Args:
        locs_file_path: Path to the .locs file
        
    Returns:
        Dictionary with channel names and their 2D/3D positions
    """
    channels = []
    x_angles = []
    y_angles = []
    
    with open(locs_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    name = parts[3]
                    
                    channels.append(name)
                    x_angles.append(x)
                    y_angles.append(y)
                except (ValueError, IndexError):
                    continue
    
    return {
        'names': channels,
        'x': np.array(x_angles),
        'y': np.array(y_angles)
    }


def angles_to_3d(theta: np.ndarray, radius: np.ndarray, head_radius: float = 1.0) -> np.ndarray:
    """
    Convert EEGLAB polar coordinates (theta, radius) to 3D Cartesian (x, y, z).
    
    given coordinate system:
    - theta (Column 2): Azimuth angle in degrees (-180 to 180, where 0° points right)
    - radius (Column 3): Polar radius (0.5 = equator, lower = top/center, higher = sides/bottom)
    
    Args:
        theta: Azimuth angle in degrees (Column 2 of .locs file)
        radius: Polar radius (Column 3 of .locs file, where 0.5 is equator)
        head_radius: Final scale of the sphere (default 1.0)
        
    Returns:
        Array of shape (n_channels, 3) with (x, y, z) coordinates
    """
    #  Azimuth to Radians
    azimuth = np.deg2rad(theta)
    
    # Polar Radius to Elevation Angle
    # EEGLAB convention:
    #   - radius 0.0 → elevation 90° (top, Cz position)
    #   - radius 0.5 → elevation 0° (equator)
    #   - radius > 0.5 → elevation < 0° (below equator)
    elevation = np.deg2rad(90 - (radius * 180))
    
    # Spherical to Cartesian Conversion
    # x = cos(elev) * sin(azim)
    # y = cos(elev) * cos(azim)
    # z = sin(elev)
    x = head_radius * np.cos(elevation) * np.sin(azimuth)
    y = head_radius * np.cos(elevation) * np.cos(azimuth)
    z = head_radius * np.sin(elevation)
    
    return np.column_stack([x, y, z])


def create_montage_from_locs(locs_file_path: str, coord_frame: str = "head") -> dict:
    """
    Create MNE montage from .locs file.
    
    Args:
        locs_file_path: Path to .locs file
        coord_frame: Coordinate frame ('head' or 'unknown')
        
    Returns:
        Dictionary with montage and channel names
    """
    locs_data = parse_locs_file(locs_file_path)
    channels = locs_data['names']
    
    # Convert .locs polar coordinates to 3D Cartesian coordinates
    positions_3d = angles_to_3d(locs_data['x'], locs_data['y'])
    
    # Create montage
    montage_dict = {ch: pos for ch, pos in zip(channels, positions_3d)}
    
    # Create MNE montage using make_dig_montage
    ch_pos = {ch: pos for ch, pos in zip(channels, positions_3d)}
    montage = make_dig_montage(ch_pos=ch_pos, coord_frame=coord_frame)
    
    return {
        'montage': montage,
        'channels': channels,
        'positions': positions_3d
    }


if __name__ == "__main__":
    # Test parsing
    locs_path = Path(__file__).parent.parent.parent / "data" / "SEED" / "channel_62_pos.locs"
    result = create_montage_from_locs(str(locs_path))
    print(f"Loaded {len(result['channels'])} channels")
    print(f"Channels: {result['channels']}")
    print(f"Montage: {result['montage']}")
