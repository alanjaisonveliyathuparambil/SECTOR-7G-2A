import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torch

from backend.pipeline import DetectionPipeline

app = FastAPI(
    title="TruthLock Deepfake Forensic Detection API",
    description="Backend AI service for real-time deepfake detection powered by Selim DFDC EfficientNet-B7 and FaceForensics++ Xception Net",
    version="1.0.0"
)

# CORS middleware for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Initialize pipeline
print("[Backend Startup] Initializing Detection Pipeline...")
pipeline = DetectionPipeline(weights_dir=WEIGHTS_DIR)
print("[Backend Startup] Detection Pipeline Ready!")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TruthLock Deepfake Detection Engine",
        "device": pipeline.engine.device
    }

@app.get("/api/models")
def get_models():
    """
    Returns information on all loaded deep learning models and forensic weights.
    """
    models_info = []
    
    # Selim DFDC
    selim_loaded = "selim_dfdc" in pipeline.engine.models_loaded
    selim_weight_file = os.path.join(WEIGHTS_DIR, "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36")
    selim_size = os.path.getsize(selim_weight_file) / (1024*1024) if os.path.exists(selim_weight_file) else 0
    models_info.append({
        "id": "selim_dfdc",
        "name": "Selim DFDC EfficientNet-B7",
        "source": "1st Place DFDC Kaggle Challenge Winner (Selim Seferbekov)",
        "backbone": "tf_efficientnet_b7_ns",
        "parameters": "66M",
        "weight_file": "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "weight_size_mb": round(selim_size, 1),
        "loaded": selim_loaded
    })

    # FaceForensics++ Xception
    xception_loaded = "faceforensics_xception" in pipeline.engine.models_loaded
    x_weight_file = os.path.join(WEIGHTS_DIR, "best_xception.pth")
    x_size = os.path.getsize(x_weight_file) / (1024*1024) if os.path.exists(x_weight_file) else 0
    models_info.append({
        "id": "faceforensics_xception",
        "name": "FaceForensics++ Xception Net",
        "source": "FaceForensics++ Deepfake Detection Benchmark",
        "backbone": "Xception CNN (Separable Convolutions)",
        "parameters": "20.8M",
        "weight_file": "best_xception.pth (Fine-tuned) & xception-b5690688.pth (Base)",
        "weight_size_mb": round(x_size, 1),
        "loaded": xception_loaded
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
    Upload an image for forensic deepfake analysis.
    Supported models: 'ensemble', 'selim_dfdc', 'faceforensics_xception'
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

@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    model: str = Form("ensemble")
):
    """
    Upload a video (.mp4, .avi, .mov, .webm) for frame-by-frame temporal forensic analysis.
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
    Returns list of preset sample media files available for instant demonstration.
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
    return {"message": "TruthLock Backend API is running. Frontend not yet compiled."}
