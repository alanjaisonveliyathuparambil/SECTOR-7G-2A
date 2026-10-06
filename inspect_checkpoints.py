import os
import torch

weights_dir = "weights"
print("Scanning weights in:", weights_dir)

for fname in os.listdir(weights_dir):
    fpath = os.path.join(weights_dir, fname)
    if not os.path.isfile(fpath) or fname.endswith(".tmp"):
        continue
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"\n--- Checking {fname} ({size_mb:.2f} MB) ---")
    try:
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            print("Type: dict, keys:", list(data.keys())[:10])
            if "state_dict" in data:
                sd = data["state_dict"]
                print("state_dict entries:", len(sd))
                print("First 3 state_dict keys:", list(sd.keys())[:3])
            else:
                print("Direct state dict entries:", len(data))
                print("First 3 keys:", list(data.keys())[:3])
        else:
            print("Loaded object type:", type(data))
    except Exception as e:
        print("Error loading:", e)
