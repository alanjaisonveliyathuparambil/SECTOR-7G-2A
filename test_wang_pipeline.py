import urllib.request
import json
import io

def test_wang_pipeline():
    # 1. Check health
    with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=10) as res:
        health = json.loads(res.read().decode("utf-8"))
        print("Health Status:", health)

    # 2. Check models endpoint
    with urllib.request.urlopen("http://127.0.0.1:8000/api/models", timeout=10) as res:
        models_data = json.loads(res.read().decode("utf-8"))
        print("\nRegistered Models:")
        for m in models_data["models"]:
            print(f"  * {m['id']}: {m['name']} (Loaded: {m['loaded']}, Size: {m['weight_size_mb']} MB)")

    # 3. Test static image upload with Wang CNNDetection model
    url = "http://127.0.0.1:8000/api/analyze/image"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    with open("samples/deepfake_synthetic_face.jpg", "rb") as f:
        img_bytes = f.read()

    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="deepfake_synthetic_face.jpg"\r\n')
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(img_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    body.write(b"wang_cnndetect\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(url, data=body.getvalue())
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    
    with urllib.request.urlopen(req, timeout=15) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        print("\n--- Wang CNNDetection Direct Analysis on Synthetic Sample ---")
        print("Status Code:", resp.status)
        print("Model Used:", res_json.get("model_used"))
        print("Overall Verdict:", res_json.get("verdict"))
        print("GAN Synthetic Footprint Score:", res_json.get("gan_synthetic_footprint_score"))
        print("Footprint Details:", json.dumps(res_json.get("gan_footprint"), indent=4))

if __name__ == "__main__":
    test_wang_pipeline()
