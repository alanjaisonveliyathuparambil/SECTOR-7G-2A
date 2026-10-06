# TRUTHLOCK AI &mdash; Deepfake & GAN Synthetic Media Forensic System

**TruthLock** is an advanced synthetic media detection web application and forensic backend powered by four pre-trained neural network architectures:
1. **OpenCLIP ViT-L-14 Diffusion Detector** &mdash; Vision Transformer backbone with UniversalFakeDetect (CVPR 2023) linear classification weights for immediate detection of Midjourney (v4/v5/v6) and Stable Diffusion (v1.5/v2.1/SDXL) imagery (`ViT-L-14.pt`, 889.6 MB; `fc_weights.pth`, 3.1 KB auto-fetched from Hugging Face).
2. **Sheng-Yu Wang CNNDetection** &mdash; Landmark CVPR 2020 architecture ("CNN-generated images are surprisingly easy to spot... for now") detecting universal CNN generator artifacts across ProGAN, StyleGAN, BigGAN, and diffusion models (`blur_jpg_prob0.5.pth`, 269.4 MB).
3. **Selim DFDC Model** &mdash; 1st-place solution in Meta's Kaggle Deepfake Detection Challenge by Selim Seferbekov (`tf_efficientnet_b7_ns` backbone, 254.5 MB).
4. **FaceForensics++ Xception Net** &mdash; Benchmark Deepfake Detection architecture (`xception-b5690688.pth` base & `best_xception.pth` classifier, 102.7 MB).

---

## Architecture Overview

```
TRUTH_LOCK/
├── weights/                     # Pre-trained neural network checkpoints
│   ├── ViT-L-14.pt                                              (889.6 MB) [OpenCLIP ViT-L-14 Backbone]
│   ├── fc_weights.pth                                           (3.1 KB)   [HF UniversalFakeDetect Probe]
│   ├── blur_jpg_prob0.5.pth                                     (269.4 MB) [Wang CNNDetection]
│   ├── final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36  (254.5 MB) [Selim DFDC]
│   ├── xception-b5690688.pth                                    (87.4 MB)  [FF++ Xception Base]
│   └── best_xception.pth                                        (98.0 MB)  [FF++ Xception Classifier]
├── models/
│   ├── clip_diffusion_detector.py # OpenCLIP ViT-L-14 & HF linear probe detector
│   ├── wang_cnndetect.py        # Sheng-Yu Wang ResNet-50 GAN detector
│   ├── selim_dfdc.py            # EfficientNet-B7 DFDC architecture
│   └── xception.py              # Xception CNN with Separable Convolutions
├── backend/
│   ├── face_detector.py         # Multi-scale facial ROI extraction with padding
│   ├── forensics.py             # Multi-domain engine (Diffusion probe, FFT 2D, Laplacian, GAN Footprint)
│   ├── pipeline.py              # Image and video timeline coordinator
│   └── app.py                   # FastAPI REST API & static file server
├── app.py                       # Root FastAPI application entrypoint
├── frontend/
│   ├── index.html               # Cyber-forensic HUD interface with OpenCLIP selector
│   ├── index.css                # Glassmorphic dark styling, Diffusion & GAN footprint cards
│   └── app.js                   # Client controller, live scanning, SVG gauges
├── samples/                     # Built-in demo files for 1-click testing
│   ├── authentic_portrait.jpg
│   └── deepfake_synthetic_face.jpg
├── requirements.txt             # PyTorch, OpenCLIP, FastAPI, OpenCV, Uvicorn, TIMM
└── test_wang_pipeline.py        # End-to-end verification script
```

---

## REST Endpoints

- `POST /api/detect/diffusion` &mdash; Evaluates image with OpenCLIP ViT-L-14 and UniversalFakeDetect linear weights; returns probability, verdict, and generator classification (Midjourney vs. Stable Diffusion).
- `POST /api/analyze/gan-footprint` &mdash; Dedicated Sheng-Yu Wang CNNDetection structural GAN evaluation.
- `POST /api/analyze/image` &mdash; Full multi-model forensic analysis (supports `model="ensemble"`, `"clip_diffusion"`, `"wang_cnndetect"`, `"selim_dfdc"`, `"faceforensics_xception"`).
- `POST /api/analyze/video` &mdash; Temporal frame-by-frame deepfake analysis.
- `GET /api/models` &mdash; Details and status of all 4 neural network models.
- `GET /api/health` &mdash; Server and loaded models health check.

---

## How to Run the Website

### 1. Start the Backend Server
Run the FastAPI server with Uvicorn:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### 2. Open in Your Browser
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in Chrome, Edge, or Firefox.
---

## Key Features

- **GAN Synthetic Footprint Score**:
  - Global structural artifact analysis via Sheng-Yu Wang's ResNet-50 model trained on ProGAN/StyleGAN with blur and JPEG augmentations.
  - Multi-patch sampling with 2D Fast Fourier Transform (FFT) spectral lattice anomaly decomposition.
- **Tri-Model Multi-Domain Ensemble**:
  - `Tri-Model Ensemble Fusion`: Weighted consensus combining Sheng-Yu Wang CNNDetection + Selim DFDC + FaceForensics++ Xception Net.
  - Individual selection for `Wang CNNDetection`, `Selim DFDC Model`, or `FaceForensics++ Xception`.
- **Image & Video Support**:
  - High-resolution static image analysis with HUD bounding box overlays.
  - Video temporal analysis with manipulation timeline chart.
- **Export Audit Dossier**: Download structured JSON reports including the GAN Synthetic Footprint Score, confidence ratings, and signal breakdowns.
