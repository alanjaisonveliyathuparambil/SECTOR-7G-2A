import torch

d = torch.load("weights/blur_jpg_prob0.5.pth", map_location="cpu", weights_only=False)
print("Top keys:", list(d.keys()) if isinstance(d, dict) else type(d))
if isinstance(d, dict):
    target = d["model"] if "model" in d else d
    print("Total layer keys:", len(target))
    print("First 5 keys:", list(target.keys())[:5])
    print("Last 5 keys:", list(target.keys())[-5:])
    if "fc.weight" in target:
        print("fc.weight shape:", target["fc.weight"].shape)
    elif "module.fc.weight" in target:
        print("module.fc.weight shape:", target["module.fc.weight"].shape)
