import urllib.request
import io
import json

def test_api():
    url = "http://127.0.0.1:8000/api/analyze/image"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    with open("samples/authentic_portrait.jpg", "rb") as f:
        img_bytes = f.read()

    body = io.BytesIO()
    # file field
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="authentic_portrait.jpg"\r\n')
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(img_bytes)
    body.write(b"\r\n")
    # model field
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    body.write(b"ensemble\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(url, data=body.getvalue())
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    
    with urllib.request.urlopen(req, timeout=15) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        print("API Test Successful! Status Code:", resp.status)
        print("Verdict:", res_json.get("verdict"))
        print("Real Probability:", res_json.get("real_probability"))
        print("Fake Probability:", res_json.get("fake_probability"))
        print("Risk Level:", res_json.get("risk_level"))
        print("Faces Detected:", res_json.get("faces_detected"))
        print("Signals:", res_json.get("aggregate_signals"))

if __name__ == "__main__":
    test_api()
