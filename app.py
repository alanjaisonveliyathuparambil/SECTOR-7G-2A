import os
import shutil
import tempfile
from typing import Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torch

from backend.pipeline import DetectionPipeline

app = FastAPI(
    title="TruthLock Ensemble Deepfake & Generative AI Forensic Detector",
    description="FastAPI Forensic Detection Server with OpenCLIP ViT-L-14 (Midjourney & Stable Diffusion), Sheng-Yu Wang CNNDetection, Selim DFDC, and FaceForensics++ Xception",
    version="1.2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Initialize Deepfake & GAN Forensic Detection Pipeline
print("=" * 65)
print("TruthLock AI: Initializing Multi-Model Forensic Pipeline")
print(f"Loading weights from: {WEIGHTS_DIR}")
pipeline = DetectionPipeline(weights_dir=WEIGHTS_DIR)
print("Models Loaded:", list(pipeline.engine.models_loaded.keys()))
print("=" * 65)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TruthLock Ensemble Deepfake Detector",
        "device": pipeline.engine.device,
        "models_loaded": list(pipeline.engine.models_loaded.keys())
    }

@app.get("/api/models")
def get_models():
    """
    Returns information on all loaded deep learning models and forensic weights.
    """
    models_info = []

    # 1. OpenCLIP ViT-L-14 Diffusion & Midjourney Detector
    clip_loaded = "clip_vit_l14" in pipeline.engine.models_loaded
    fc_path = os.path.join(WEIGHTS_DIR, "fc_weights.pth")
    vit_path = os.path.join(WEIGHTS_DIR, "ViT-L-14.pt")
    vit_size = (os.path.getsize(vit_path) if os.path.exists(vit_path) else 0) / (1024 * 1024)
    fc_size = (os.path.getsize(fc_path) if os.path.exists(fc_path) else 0) / 1024
    models_info.append({
        "id": "clip_diffusion",
        "name": "OpenCLIP ViT-L-14 (Diffusion & Midjourney Detector)",
        "source": "OpenCLIP + UniversalFakeDetect (CVPR 2023 linear probe, Ojha et al.)",
        "backbone": "ViT-L-14 (OpenAI pre-trained, 768-dim CLIP visual embedding)",
        "parameters": "428M",
        "weight_file": "ViT-L-14.pt & fc_weights.pth",
        "weight_size_mb": round(vit_size, 1),
        "probe_size_kb": round(fc_size, 1),
        "loaded": clip_loaded,
        "specialty": "Detects Midjourney (v4/v5/v6), Stable Diffusion (v1.5/v2.1/SDXL), and DALL-E Latent Generations"
    })
    
    # 2. Sheng-Yu Wang CNNDetection
    wang_loaded = "wang_cnndetect" in pipeline.engine.models_loaded
    wang_path = os.path.join(WEIGHTS_DIR, "blur_jpg_prob0.5.pth")
    wang_size = os.path.getsize(wang_path) / (1024 * 1024) if os.path.exists(wang_path) else 0
    models_info.append({
        "id": "wang_cnndetect",
        "name": "Sheng-Yu Wang CNNDetection (ProGAN / StyleGAN)",
        "source": "CVPR 2020: 'CNN-generated images are surprisingly easy to spot... for now' (Wang et al.)",
        "backbone": "ResNet-50 with Blur & JPEG Augmentation",
        "parameters": "25.6M",
        "weight_file": "blur_jpg_prob0.5.pth",
        "weight_size_mb": round(wang_size, 1),
        "loaded": wang_loaded,
        "specialty": "Universal GAN Structural Artifacts & Lattice Generator Footprint"
    })

    # 3. Selim DFDC EfficientNet-B7
    selim_loaded = "selim_dfdc" in pipeline.engine.models_loaded
    selim_path = os.path.join(WEIGHTS_DIR, "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36")
    selim_size = os.path.getsize(selim_path) / (1024 * 1024) if os.path.exists(selim_path) else 0
    models_info.append({
        "id": "selim_dfdc",
        "name": "Selim DFDC EfficientNet-B7",
        "source": "1st Place DFDC Kaggle Competition Winner (Selim Seferbekov)",
        "backbone": "tf_efficientnet_b7_ns",
        "parameters": "66M",
        "weight_file": "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "weight_size_mb": round(selim_size, 1),
        "loaded": selim_loaded,
        "specialty": "High-Capacity Deepfake Face Swap & Warping Detection"
    })

    # 4. FaceForensics++ Xception Net
    xception_loaded = "faceforensics_xception" in pipeline.engine.models_loaded
    x_path = os.path.join(WEIGHTS_DIR, "best_xception.pth")
    x_size = os.path.getsize(x_path) / (1024 * 1024) if os.path.exists(x_path) else 0
    models_info.append({
        "id": "faceforensics_xception",
        "name": "FaceForensics++ Xception Net",
        "source": "FaceForensics++ Deepfake Detection Benchmark",
        "backbone": "Xception CNN (Separable Convolutions)",
        "parameters": "20.8M",
        "weight_file": "best_xception.pth & xception-b5690688.pth",
        "weight_size_mb": round(x_size, 1),
        "loaded": xception_loaded,
        "specialty": "Face2Face, Deepfakes, and NeuralTextures Benchmark Classifier"
    })

    return {
        "device": pipeline.engine.device,
        "cuda_available": torch.cuda.is_available(),
        "models": models_info,
        "default_mode": "ensemble"
    }

@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    model: str = Form("ensemble")
):
    """
    Forensic deepfake, GAN structural, and OpenCLIP diffusion analysis on an uploaded image.
    Model options: 'ensemble', 'clip_diffusion', 'wang_cnndetect', 'selim_dfdc', 'faceforensics_xception'
    Returns comprehensive multi-model verdict and synthetic footprint scores.
    """
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        result = pipeline.analyze_image_bytes(contents, model_name=model)
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/diffusion")
@app.post("/api/analyze/diffusion")
async def detect_diffusion(file: UploadFile = File(...)):
    """
    Image detection route using pre-trained 'ViT-L-14' OpenCLIP backbone
    with baseline open-source linear classification weights (UniversalFakeDetect)
    specifically tuned to detect Midjourney and Stable Diffusion images.
    """
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        import cv2
        import numpy as np
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")
            
        diff_result = pipeline.engine.evaluate_diffusion_image(img)
        diff_result["filename"] = file.filename
        return JSONResponse(content=diff_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/gan-footprint")
async def analyze_gan_footprint(file: UploadFile = File(...)):
    """
    Dedicated endpoint evaluating static image uploads strictly for
    Sheng-Yu Wang's GAN structural artifacts and synthetic footprint score.
    """
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        import cv2
        import numpy as np
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")
            
        footprint = pipeline.engine.evaluate_gan_footprint(img)
        footprint["filename"] = file.filename
        return JSONResponse(content=footprint)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    model: str = Form("ensemble")
):
    """
    Frame-by-frame temporal forensic analysis on an uploaded video (.mp4, .avi, .mov).
    """
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = pipeline.analyze_video_file(temp_video_path, model_name=model, max_frames=16)
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/api/samples")
def list_samples():
    """
    List preset sample portraits and test images.
    """
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for f in os.listdir(SAMPLES_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4')):
                is_fake = "fake" in f.lower() or "deepfake" in f.lower()
                samples.append({
                    "name": f,
                    "type": "video" if f.endswith('.mp4') else "image",
                    "expected_type": "Synthetic Deepfake" if is_fake else "Authentic Portrait",
                    "url": f"/api/sample-media/{f}"
                })
    return {"samples": samples}

@app.get("/api/sample-media/{filename}")
def get_sample_media(filename: str):
    fpath = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Sample not found.")
    return FileResponse(fpath)

# Static frontend files mount
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "TruthLock Ensemble API is live. Frontend static file index.html not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
