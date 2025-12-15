"""
Check what standard positions REVE actually has.
"""

from transformers import AutoModel
import json
from pathlib import Path

# Load REVE's position bank
try:
    print("Loading REVE position bank...")
    pos_bank = AutoModel.from_pretrained('brain-bzh/reve-positions', trust_remote_code=True)
    
    # Check what positions it recognizes
    print("\nREVE position bank attributes:")
    print(dir(pos_bank))
    
    # Try to get position information
    print("\n" + "="*80)
    print("Checking for position data in pos_bank...")
    print("="*80)
    
    # Check if there's position information in the config or model
    if hasattr(pos_bank, 'config'):
        print("\nConfig attributes:")
        print(dir(pos_bank.config))
        if hasattr(pos_bank.config, 'position_names'):
            print(f"Position names: {pos_bank.config.position_names}")
    
    # Try calling with common positions to see what works
    test_positions = ['Fp1', 'Fp2', 'Cz', 'Pz', 'O1', 'O2']
    try:
        print(f"\nTesting with positions: {test_positions}")
        result = pos_bank(test_positions)
        print(f"✓ Successfully got embeddings for {len(test_positions)} positions")
        print(f"Embedding shape: {result.shape if hasattr(result, 'shape') else type(result)}")
    except Exception as e:
        print(f"Error with test positions: {e}")
    
    # Check for position metadata
    print("\n" + "="*80)
    print("Checking model structure...")
    print("="*80)
    
    if hasattr(pos_bank, 'position_embeddings'):
        pos_embeddings = pos_bank.position_embeddings
        print(f"\nPosition embeddings found: {type(pos_embeddings)}")
        print(f"Num embeddings: {pos_embeddings.num_embeddings if hasattr(pos_embeddings, 'num_embeddings') else 'unknown'}")
        print(f"Embedding dim: {pos_embeddings.embedding_dim if hasattr(pos_embeddings, 'embedding_dim') else 'unknown'}")
    
    # Check state dict for position names
    if hasattr(pos_bank, 'state_dict'):
        print("\nModel state dict keys (first 20):")
        for i, key in enumerate(list(pos_bank.state_dict().keys())[:20]):
            print(f"  {key}")
    
except Exception as e:
    print(f"Error loading REVE: {e}")
    print("\nTrying alternative approaches...")
    
    # Try to load from cache
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    if cache_dir.exists():
        print(f"\nCache directory: {cache_dir}")
        for item in list(cache_dir.iterdir())[:5]:
            print(f"  {item.name}")
