# REVE Model: Using Actual 3D Electrode Coordinates

## Overview

The REVE paper's key contribution is a **4D positional encoding** that enables the model to handle **arbitrary electrode configurations** without relying on learned position embeddings. Your implementation should leverage this flexibility by passing actual measured electrode coordinates.

From the paper:
> "our method uses a transformation applicable to each position, utilizing the actual 3D coordinates and timestep of each EEG patch, enabling the model to handle arbitrary electrode configurations and sequence lengths without relying on learned embeddings."

## REVE's 4D Positional Encoding

### What is it?

The REVE model uses a 4D positional encoding that combines:

1. **Temporal Dimension**: The timestep `t` (position in the sequence)
2. **Spatial 3D Coordinates**: (x, y, z) from your electrode_positions.json

### Formula (Conceptual)

```
PE(t, x, y, z) = combination of sinusoidal functions
                  with temporal and spatial frequencies
```

This is similar to standard transformer positional encoding but extended to 3D space + time.

### Why is this important?

- **Learns position relationships, not specific montages**: The model learns how to interpret spatial and temporal patterns, not memorizes "electrode 5 is at position X"
- **Generalizes to different electrode layouts**: If trained on one montage, it can adapt to another
- **No position embedding lookup**: You pass raw coordinates, not learned embeddings

## How to Use in Your Code

### Step 1: Load Actual 3D Coordinates

Your electrode_positions.json has format:
```json
{
  "1": {"x": -86.19, "y": 2.54, "z": 28.02, "label": "1"},
  "2": {"x": -79.67, "y": 44.83, "z": 54.12, "label": "2"},
  ...
  "96": {...}
}
```

This is loaded in `feature_extraction.py`:
```python
def _load_electrode_coordinates(self):
    """Load actual 3D electrode coordinates from the XDF extraction."""
    coords_path = Path(__file__).parent / "electrodes_pos" / "electrode_positions.json"
    
    with open(coords_path, 'r') as f:
        positions_dict = json.load(f)
    
    # Convert to numpy array: shape (96, 3) with x, y, z for each electrode
    self.electrode_coordinates = np.zeros((96, 3))
    
    for electrode_num in range(1, 97):
        key = str(electrode_num)
        if key in positions_dict:
            coord = positions_dict[key]
            self.electrode_coordinates[electrode_num - 1] = [coord['x'], coord['y'], coord['z']]
```

**Result**: `self.electrode_coordinates` shape is `(96, 3)`

### Step 2: Pass Coordinates to Model Setup

In `feature_extraction.py::load_model()`:
```python
self.model, self.positions = setup_model(
    self.model, 
    self.pos_bank, 
    num_classes=num_classes,
    electrode_coordinates=self.electrode_coordinates  # ← Pass actual 3D coords
)
```

In `reve_model.py::setup_model()`:
```python
if electrode_coordinates is not None:
    # Select only the channels you actually use
    positions_array = electrode_coordinates[:NUM_CHANNELS]
    
    # Convert to tensor
    positions = torch.from_numpy(positions_array).float()
    
    # Now positions has shape (num_channels, 3) with actual 3D coordinates
    return model, positions
```

**Result**: `self.positions` shape is `(num_channels, 3)`

### Step 3: Pass Positions During Forward Pass

This is the **CRITICAL PART**. In `feature_extraction.py::extract_features()`:

```python
with torch.no_grad():
    for i in range(0, num_epochs, batch_size):
        batch = epochs_tensor[i:batch_end].to(self.device)
        
        # CRITICAL: Pass positions (3D coordinates) to REVE
        if self.positions is not None:
            pos = self.positions.to(self.device)
            
            # REVE forward pass: model(eeg_signals, positions)
            output = self.model(batch, pos)  # ← This is key!
        
        # Extract features
        features = output.reshape(output.shape[0], -1)
        all_features.append(features.cpu().numpy())
```

**Key Points**:
- `batch` shape: `(batch_size, num_channels, sequence_length)`
- `pos` shape: `(num_channels, 3)` - the actual 3D coordinates
- The REVE model uses these coordinates to compute 4D positional encoding internally
- You pass the same `pos` for all batches (coordinates don't change)

## Coordinate System

Your CapTrak coordinates are in **millimeters** with the origin at the head center:
- **X**: Lateral (left-right), negative is left, positive is right
- **Y**: Anterior-posterior (front-back), negative is back, positive is front  
- **Z**: Vertical (up-down), negative is down, positive is up

From `electrode_positions.json`:
- **X range**: [-86.19, 86.19] mm (roughly head width ~17 cm)
- **Y range**: [-76.31, 87.21] mm (roughly head front-back ~16 cm)
- **Z range**: [-23.78, 130.74] mm (roughly head height ~15 cm)

These ranges make sense for a human head!

## What NOT to Do

❌ **Don't use position embeddings from pos_bank** if you have actual coordinates
- The pos_bank gives learned embeddings (not the same as coordinates)
- These assume a standard 10-20 layout
- Your actual coordinates are more accurate

❌ **Don't reshape positions incorrectly**
- Wrong: `pos.repeat(batch_size, 1, 1)` → Creates duplicate position information
- Correct: `pos.to(device)` → Coordinates are shared across batch

❌ **Don't forget to pass positions to the model**
- The REVE model REQUIRES positions for 4D encoding
- Without them, the 4D encoding can't be computed

## Example Usage Flow

```python
# 1. Initialize feature extractor (loads coordinates automatically)
extractor = REVEFeatureExtractor()

# 2. Load model (passes coordinates to setup_model)
extractor.load_model()
# → self.electrode_coordinates: shape (96, 3)
# → self.positions: shape (num_channels, 3)

# 3. Extract features (passes coordinates to forward pass)
features = extractor.extract_features(epochs)
# → For each batch, REVE computes 4D encoding using:
#   - Temporal: sequence positions
#   - Spatial: actual 3D coordinates from electrode_positions.json

# 4. Features are extracted with awareness of electrode geometry!
```

## Benefits

1. **Handles arbitrary montages**: If you change which electrodes you use, the model can adapt
2. **Leverages real geometry**: Uses measured electrode positions, not standard 10-20
3. **No position embedding lookup**: More efficient, no learned position tables
4. **Generalizable**: Model learns spatial patterns, not memorizes specific layouts

## Troubleshooting

### Issue: "Positions not loaded!"
**Cause**: `self.positions` is None
**Solution**: Ensure `_load_electrode_coordinates()` returns True and coordinates are passed to `setup_model()`

### Issue: Shape mismatch in forward pass
**Cause**: Positions shape wrong (e.g., `(96, hidden_dim)` instead of `(num_channels, 3)`)
**Solution**: Check that `setup_model()` returns raw 3D coordinates, not embeddings

### Issue: Model doesn't generalize across different electrode sets
**Cause**: Using position embeddings instead of coordinates
**Solution**: Make sure `electrode_coordinates` parameter is passed (not None)

## References

- REVE Paper: [Link to paper]
- REVE Implementation: brain-bzh/reve-base-model (HuggingFace)
- Your data: CapTrak electrode positions from XDF metadata
