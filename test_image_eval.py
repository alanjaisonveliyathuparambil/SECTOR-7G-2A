import os
import cv2
import numpy as np
from backend.pipeline import DetectionPipeline

pipeline = DetectionPipeline(weights_dir="weights")

print("\n--- Testing samples/authentic_portrait.jpg ---")
with open("samples/authentic_portrait.jpg", "rb") as f:
    res = pipeline.analyze_image_bytes(f.read(), filename="authentic_portrait.jpg")

print(f"Overall Fake Prob: {res.get('fake_probability')}")
print(f"Verdict: {res.get('verdict')}")
print(f"Risk: {res.get('risk_level')}")
print(f"Is Compromised: {res.get('is_compromised')}")
print(f"Manipulation Type: {res.get('manipulation_type')}")
for layer_id, l in res.get("layers", {}).items():
    print(f"  {l['layer_name']}: score={l['score']}, is_flagged={l['is_flagged']}, status={l['status']}")
    if "metrics" in l:
        print(f"    metrics: {l['metrics']}")

print("\n--- Testing samples/ai_generated_portrait.jpg ---")
with open("samples/ai_generated_portrait.jpg", "rb") as f:
    res_ai = pipeline.analyze_image_bytes(f.read(), filename="ai_generated_portrait.jpg")

print(f"Overall Fake Prob: {res_ai.get('fake_probability')}")
print(f"Verdict: {res_ai.get('verdict')}")
print(f"Risk: {res_ai.get('risk_level')}")
print(f"Is Compromised: {res_ai.get('is_compromised')}")
print(f"Manipulation Type: {res_ai.get('manipulation_type')}")
for layer_id, l in res_ai.get("layers", {}).items():
    print(f"  {l['layer_name']}: score={l['score']}, is_flagged={l['is_flagged']}, status={l['status']}")

print("\n--- Testing samples/morphed_face_sample.jpg ---")
with open("samples/morphed_face_sample.jpg", "rb") as f:
    res_morph = pipeline.analyze_image_bytes(f.read(), filename="morphed_face_sample.jpg")

print(f"Overall Fake Prob: {res_morph.get('fake_probability')}")
print(f"Verdict: {res_morph.get('verdict')}")
print(f"Risk: {res_morph.get('risk_level')}")
print(f"Is Compromised: {res_morph.get('is_compromised')}")
print(f"Manipulation Type: {res_morph.get('manipulation_type')}")
for layer_id, l in res_morph.get("layers", {}).items():
    print(f"  {l['layer_name']}: score={l['score']}, is_flagged={l['is_flagged']}, status={l['status']}")

print("\n--- Testing samples/deepfake_synthetic_face.jpg ---")
with open("samples/deepfake_synthetic_face.jpg", "rb") as f:
    res_deep = pipeline.analyze_image_bytes(f.read(), filename="deepfake_synthetic_face.jpg")

print(f"Overall Fake Prob: {res_deep.get('fake_probability')}")
print(f"Verdict: {res_deep.get('verdict')}")
print(f"Risk: {res_deep.get('risk_level')}")
print(f"Is Compromised: {res_deep.get('is_compromised')}")
print(f"Manipulation Type: {res_deep.get('manipulation_type')}")
for layer_id, l in res_deep.get("layers", {}).items():
    print(f"  {l['layer_name']}: score={l['score']}, is_flagged={l['is_flagged']}, status={l['status']}")

print("\n--- Testing Compressed Authentic Photo (JPEG 85) ---")
img_orig = cv2.imread("samples/authentic_portrait.jpg")
_, enc85 = cv2.imencode(".jpg", img_orig, [cv2.IMWRITE_JPEG_QUALITY, 85])
res_j85 = pipeline.analyze_image_bytes(enc85.tobytes(), filename="authentic_camera_photo.jpg")
print(f"Overall Fake Prob: {res_j85.get('fake_probability')}")
print(f"Verdict: {res_j85.get('verdict')}")
print(f"Risk: {res_j85.get('risk_level')}")
print(f"Is Compromised: {res_j85.get('is_compromised')}")
print(f"Manipulation Type: {res_j85.get('manipulation_type')}")
for layer_id, l in res_j85.get("layers", {}).items():
    print(f"  {l['layer_name']}: score={l['score']}, is_flagged={l['is_flagged']}, status={l['status']}")
