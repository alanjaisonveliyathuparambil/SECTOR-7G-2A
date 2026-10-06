import os, cv2, torch
import numpy as np
from PIL import Image
from backend.pipeline import DetectionPipeline

pipe = DetectionPipeline("weights")

test_files = [
    ("samples/authentic_portrait.jpg", "AUTHENTIC"),
    ("samples/authentic_camera_photo.jpg", "AUTHENTIC"),
    ("samples/ai_generated_portrait.jpg", "SYNTHETIC"),
    ("samples/morphed_face_sample.jpg", "SYNTHETIC"),
    ("samples/deepfake_synthetic_face.jpg", "SYNTHETIC"),
    (r"C:\Users\REJOICE\OneDrive\Pictures\Camera Roll\WIN_20261006_01_32_23_Pro.jpg", "AUTHENTIC")
]

print("\n================== FULL 6-IMAGE SUITE TEST ==================")
for path, expected in test_files:
    fname = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    res = pipe.analyze_image_bytes(data, filename=fname)
    verdict = res.get("verdict")
    prob = res.get("fake_probability")
    risk = res.get("risk_level")
    is_comp = res.get("is_compromised")
    manip = res.get("manipulation_type")
    
    # Layer scores
    l1 = res["layers"]["layer1_spatial_boundaries"]["score"]
    l2 = res["layers"]["layer2_depthwise_compression"]["score"]
    l3 = res["layers"]["layer3_frequency_upsampling"]["score"]
    l4 = res["layers"]["layer4_semantic_contrast"]["score"]
    
    status = "PASS" if verdict == expected else "FAIL"
    print(f"[{status}] {fname:30s} -> Verdict: {verdict:9s} (Expected: {expected:9s}) | Prob: {prob*100:5.1f}% | L1={l1*100:.1f}%, L2={l2*100:.1f}%, L3={l3*100:.1f}%, L4={l4*100:.1f}% | Type: {manip}")
