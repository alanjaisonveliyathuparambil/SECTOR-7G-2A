import os
import sys
import time
import urllib.request

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MODELS = [
    {
        "name": "Selim DFDC EfficientNet-B7",
        "filename": "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "expected_min_bytes": 200 * 1024 * 1024,  # ~266 MB
    },
    {
        "name": "FaceForensics++ Xception Base",
        "filename": "xception-b5690688.pth",
        "url": "https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth",
        "expected_min_bytes": 80 * 1024 * 1024,  # ~91 MB
    },
    {
        "name": "FaceForensics / Deepfake Xception Classifier",
        "filename": "best_xception.pth",
        "url": "https://huggingface.co/RamadhanZome/deepfake-xception/resolve/main/best_xception.pth",
        "expected_min_bytes": 80 * 1024 * 1024,  # ~102 MB
    }
]

def download_file(url: str, dest_path: str, model_name: str, min_size: int):
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size >= min_size:
            print(f"[FOUND] {model_name} already exists ({size / (1024*1024):.1f} MB). Skipping download.")
            return True
        else:
            print(f"[RETRY] {model_name} existing file is incomplete ({size} bytes). Re-downloading...")
            try:
                os.remove(dest_path)
            except Exception:
                pass

    print(f"\n[DOWNLOADING] {model_name}...")
    print(f"  Source: {url}")
    print(f"  Target: {dest_path}")

    temp_path = dest_path + ".tmp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    last_print = start_time

    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as out_file:
            total_size = resp.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else 0
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 2.0 or downloaded == total_bytes:
                    last_print = now
                    elapsed = max(now - start_time, 0.001)
                    speed = (downloaded / (1024 * 1024)) / elapsed
                    if total_bytes > 0:
                        pct = (downloaded / total_bytes) * 100
                        print(f"  Progress: {pct:5.1f}% | {downloaded / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB | {speed:.2f} MB/s", flush=True)
                    else:
                        print(f"  Downloaded: {downloaded / (1024*1024):.1f} MB | {speed:.2f} MB/s", flush=True)

        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        final_size = os.path.getsize(dest_path)
        print(f"[COMPLETED] {model_name} successfully saved ({final_size / (1024*1024):.1f} MB).\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed downloading {model_name}: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False

def main():
    print("=" * 65)
    print("TruthLock Pre-trained Weights Downloader")
    print("Target Directory:", WEIGHTS_DIR)
    print("=" * 65)

    success_count = 0
    for model in MODELS:
        dest = os.path.join(WEIGHTS_DIR, model["filename"])
        ok = download_file(model["url"], dest, model["name"], model["expected_min_bytes"])
        if ok:
            success_count += 1

    print("=" * 65)
    print(f"Download Finished: {success_count}/{len(MODELS)} model weight files ready in {WEIGHTS_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
