import os, io, json, mimetypes, uuid, urllib.request, cv2

def post_multipart(url, file_path, field_name="file"):
    boundary = uuid.uuid4().hex
    filename = os.path.basename(file_path)
    
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(file_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

url = "http://127.0.0.1:8000/api/analyze/image"

# Create compressed real camera photo
orig = cv2.imread("samples/authentic_portrait.jpg")
_, enc85 = cv2.imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, 85])
with open("samples/authentic_camera_photo.jpg", "wb") as f:
    f.write(enc85.tobytes())

test_files = [
    ("samples/authentic_portrait.jpg", "AUTHENTIC"),
    ("samples/authentic_camera_photo.jpg", "AUTHENTIC"),
    ("samples/ai_generated_portrait.jpg", "SYNTHETIC"),
    ("samples/morphed_face_sample.jpg", "SYNTHETIC"),
    ("samples/deepfake_synthetic_face.jpg", "SYNTHETIC")
]

print("\n================== LIVE API TEST SUITE ==================")
all_pass = True
for fpath, expected_verdict in test_files:
    fname = os.path.basename(fpath)
    res = post_multipart(url, fpath)
    verdict = res.get("verdict")
    is_comp = res.get("is_compromised")
    prob = res.get("fake_probability")
    risk = res.get("risk_level")
    manip = res.get("manipulation_type")
    
    status = "PASS" if verdict == expected_verdict else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"[{status}] {fname:30s} -> Verdict: {verdict:9s} (Expected: {expected_verdict:9s}) | Prob: {prob*100:.1f}% | Risk: {risk} | Compromised: {is_comp} | Type: {manip}")
print("=========================================================")
print(f"OVERALL STATUS: {'ALL TESTS PASSED WITH 100% ACCURACY!' if all_pass else 'FAILURES DETECTED!'}")
