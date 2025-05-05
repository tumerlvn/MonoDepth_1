import torch

checkpoint = torch.load("Depth-Anything-V2/checkpoints/latest25october.pth", map_location='cpu')
print("Checkpoint keys:", checkpoint.keys())  # See available keys