"""
Quick test to verify REVE model loading works
"""

from src.reve_model import load_reve_model

if __name__ == "__main__":
    print("Testing REVE model loader...")
    
    # Try loading model
    try:
        model = load_reve_model(device="cpu")
        print("✓ Model loaded successfully!")
        print(f"Model type: {type(model)}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
