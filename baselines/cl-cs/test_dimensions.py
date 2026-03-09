"""
Dimension verification script for the THU-EP refactored ConvNet_baseNonlinearHead.

Expected tensor flow for input (B, 1, 30, 1250):
  spatialConv  (30,1)       -> (B, 16, 1,    1250)
  permute                   -> (B, 1,  16,   1250)
  timeConv     (1,60,p=29)  -> (B, 16, 16,   1249)
  frequencyConv(1,60,s=5)   -> (B, 16, 16,   238)
  AvgPool      (1,30)       -> (B, 16, 16,   7)
  spatialConv2 (16,1)       -> (B, 32, 1,    7)
  timeConv2    (1,4)        -> (B, 64, 1,    4)
  frequencyConv2 (1,3)      -> (B, 64, 1,    2)
  reshape                   -> (B, 128)
"""
import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import ConvNet_baseNonlinearHead

args = argparse.Namespace(device=torch.device('cpu'))

batch_size = 4
model = ConvNet_baseNonlinearHead(
    n_spatialFilters=16,
    n_timeFilters=16,
    timeFilterLen=60,
    n_channs=30,
    stratified=[],        # no stratified norm so shape doesn't depend on batch structure
    multiFact=2,
    isMaxPool=False,
    args=args,
)
model.eval()

x = torch.randn(batch_size, 1, 30, 1250)
print(f"Input shape:  {x.shape}")

with torch.no_grad():
    out = model(x)

print(f"Output shape: {out.shape}")
assert out.shape == (batch_size, 128), (
    f"[FAIL] Expected output shape (batch={batch_size}, 128), got {out.shape}"
)
print(f"[PASS] Forward pass succeeded. Output shape: {out.shape}")
