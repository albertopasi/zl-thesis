#!/usr/bin/env python
"""
Extract electrode positions from CapTrak metadata in XDF file and generate electrode_positions.json
with only the coordinates present in the XDF file (no post-processing or calculations).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pyxdf

def extract_from_xml(xml_string):
    """Parse CapTrak XML and extract electrode positions (Name, X, Y, Z for electrodes 1-96)"""
    root = ET.fromstring(xml_string)
    electrodes_elem = root.find('.//CapTrakElectrodeList')
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
            if 1 <= electrode_num <= 96:
                positions[electrode_num] = {
                    'x': float(x_elem.text),
                    'y': float(y_elem.text),
                    'z': float(z_elem.text),
                    'label': name_elem.text
                }
        except (ValueError, TypeError):
            pass  # Skip non-numeric names (Nasion, LPA, RPA, GND)
    
    return positions

def main():
    """
    Main pipeline:
    1. Load XDF file
    2. Find CapTrak stream and extract XML metadata
    3. Parse CapTrak XML to extract electrode coordinates
    4. Save to electrode_positions.json
    """
    
    print("=" * 80)
    print("EXTRACTING ELECTRODE POSITIONS FROM XDF FILE")
    print("=" * 80)
    
    # Find XDF file
    xdf_dir = Path(__file__).parent.parent.parent.parent / 'data'
    xdf_files = list(xdf_dir.glob('**/sub-*_eeg.xdf'))
    
    if not xdf_files:
        print("ERROR: No XDF files found in data directory")
        return
    
    xdf_file = xdf_files[0]
    print(f"\nLoading: {xdf_file}")
    
    # Load XDF
    streams, header = pyxdf.load_xdf(str(xdf_file))
    print(f"Found {len(streams)} streams")
    
    # Find CapTrak stream
    captrak_stream = None
    for stream in streams:
        if stream['info']['name'][0] == 'captrak':
            captrak_stream = stream
            break
    
    if not captrak_stream:
        print("ERROR: CapTrak stream not found")
        return
    
    # Extract XML from CapTrak metadata
    print("\nParsing CapTrak XML metadata...")
    xml_string = captrak_stream['info']['desc'][0]['captrak'][0]['full_metadata'][0]
    
    # Extract coordinates from XML
    positions = extract_from_xml(xml_string)
    print(f"Found {len(positions)} electrodes in CapTrak XML")
    
    # Save to JSON
    output_file = Path(__file__).parent.parent / 'electrodes_pos' / 'electrode_positions.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(positions, f, indent=2)
    
    print(f"\n✓ Extracted {len(positions)} electrode positions (1-96) from XDF")
    print(f"✓ Saved to: {output_file}")
    
    # Show sample electrodes
    print(f"\nSample electrodes:")
    for elec_id in ['1', '25', '96']:
        if elec_id in positions:
            p = positions[elec_id]
            print(f"  Electrode {elec_id}: x={p['x']:8.2f}, y={p['y']:8.2f}, z={p['z']:8.2f} mm")

if __name__ == '__main__':
    main()
