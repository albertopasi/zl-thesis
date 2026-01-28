"""
Electrode position extraction from XDF CapTrak metadata.

Handles parsing and extraction of 3D electrode coordinates from EEG recording files.
"""

from typing import Dict
import numpy as np
import pyxdf
from pathlib import Path
import xml.etree.ElementTree as ET


class ElectrodePositionExtractor:
    """Extract 3D electrode positions from XDF CapTrak metadata."""
    
    @staticmethod
    def extract_from_xdf(xdf_path: str) -> np.ndarray:
        """
        Extract electrode positions from XDF file CapTrak metadata.
        
        Args:
            xdf_path: Path to XDF file
            
        Returns:
            np.ndarray: Electrode positions of shape (n_electrodes, 3) with x, y, z coordinates
            
        Raises:
            FileNotFoundError: If XDF file or CapTrak stream not found
            ValueError: If electrode coordinates cannot be extracted
        """
        xdf_path = Path(xdf_path)
        if not xdf_path.exists():
            raise FileNotFoundError(f"XDF file not found: {xdf_path}")
        
        print(f"\nExtracting electrode positions from: {xdf_path.name}")
        
        # Load XDF file
        streams, header = pyxdf.load_xdf(str(xdf_path))
        
        # Find CapTrak stream
        captrak_stream = None
        for stream in streams:
            if stream['info']['name'][0] == 'captrak':
                captrak_stream = stream
                break
        
        if not captrak_stream:
            raise ValueError("CapTrak stream not found in XDF file")
        
        # Extract XML from CapTrak metadata
        try:
            xml_string = captrak_stream['info']['desc'][0]['captrak'][0]['full_metadata'][0]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Could not extract CapTrak XML metadata: {e}")
        
        # Parse electrode positions from XML
        positions = parse_captrak_xml(xml_string)
        
        if not positions:
            raise ValueError("No electrode positions found in CapTrak XML")
        
        # Convert to numpy array (96 electrodes x 3 coordinates)
        n_electrodes = max(pos_dict.get('electrode_id', 0) for pos_dict in positions.values())
        electrode_array = np.zeros((n_electrodes, 3))
        
        for electrode_id, coords in positions.items():
            idx = int(electrode_id) - 1  # Convert to 0-based index
            if 0 <= idx < n_electrodes:
                electrode_array[idx] = [coords['x'], coords['y'], coords['z']]
        
        print(f"[OK] Extracted {len(positions)} electrode positions")
        print(f"  Coordinate ranges:")
        print(f"    X: [{electrode_array[:, 0].min():.2f}, {electrode_array[:, 0].max():.2f}] mm")
        print(f"    Y: [{electrode_array[:, 1].min():.2f}, {electrode_array[:, 1].max():.2f}] mm")
        print(f"    Z: [{electrode_array[:, 2].min():.2f}, {electrode_array[:, 2].max():.2f}] mm")
        
        return electrode_array


def parse_captrak_xml(xml_string: str) -> Dict[str, Dict]:
    """
    Parse CapTrak XML and extract electrode positions.
    
    Args:
        xml_string: XML string from CapTrak metadata
        
    Returns:
        dict: Dictionary mapping electrode_id to {x, y, z} coordinates
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse CapTrak XML: {e}")
    
    electrodes_elem = root.find('.//CapTrakElectrodeList')
    if electrodes_elem is None:
        raise ValueError("CapTrakElectrodeList not found in XML")
    
    electrode_elements = electrodes_elem.findall('CapTrakElectrode')
    
    positions = {}
    for el_elem in electrode_elements:
        name_elem = el_elem.find('Name')
        x_elem = el_elem.find('X')
        y_elem = el_elem.find('Y')
        z_elem = el_elem.find('Z')
        
        if None in [name_elem, x_elem, y_elem, z_elem]:
            continue
        
        try:
            electrode_num = int(name_elem.text)
            # Only keep electrodes 1-96
            if 1 <= electrode_num <= 96:
                positions[str(electrode_num)] = {
                    'electrode_id': electrode_num,
                    'x': float(x_elem.text),
                    'y': float(y_elem.text),
                    'z': float(z_elem.text),
                }
        except (ValueError, TypeError):
            # Skip non-numeric names (Nasion, LPA, RPA, GND, etc.)
            pass
    
    return positions
