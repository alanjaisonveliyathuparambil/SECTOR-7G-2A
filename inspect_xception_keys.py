import torch
import torch.nn as nn

sd = torch.load('weights/best_xception.pth', map_location='cpu', weights_only=False)
print("Total keys:", len(sd))
print("First 20 keys:")
for k in list(sd.keys())[:20]:
    print(" ", k, sd[k].shape)
print("\nMiddle keys:")
for k in list(sd.keys())[50:65]:
    print(" ", k, sd[k].shape)
print("\nLast 15 keys:")
for k in list(sd.keys())[-15:]:
    print(" ", k, sd[k].shape)
