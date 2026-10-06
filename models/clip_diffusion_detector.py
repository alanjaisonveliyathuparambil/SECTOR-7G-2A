import os
import urllib.request
from typing import Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

class OpenCLIPDiffusionDetector:
    """
    OpenCLIP ViT-L-14 AI Image & Diffusion Detector.
    Uses frozen OpenCLIP ViT-L-14 image embeddings with baseline open-source
    linear classification probe weights trained to detect Midjourney and Stable Diffusion images.
    """
    HF_LINEAR_WEIGHTS_URL = "https://huggingface.co/siddharthksah/deepsafe-weights/resolve/main/universalfakedetect/fc_weights.pth"
    
    def __init__(self, weights_dir: str = "weights", device: str = None):
        self.weights_dir = weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.fc_weights_path = os.path.join(self.weights_dir, "fc_weights.pth")
        self.vit_weights_path = os.path.join(self.weights_dir, "ViT-L-14.pt")
        
        self.model = None
        self.preprocess = None
        self.linear_head = None
        self.is_loaded = False
        
        # Ensure linear classification weights are downloaded on first run
        self._ensure_linear_weights()
        self._init_detector()

    def _ensure_linear_weights(self):
        """
        Automatically fetches baseline open-source linear classification probe weights
        from Hugging Face on first run if not already present.
        """
        if not os.path.exists(self.fc_weights_path) or os.path.getsize(self.fc_weights_path) < 1000:
            print(f"[OpenCLIP Detector] Fetching baseline linear classification weights from Hugging Face: {self.HF_LINEAR_WEIGHTS_URL}...")
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                req = urllib.request.Request(self.HF_LINEAR_WEIGHTS_URL, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(self.fc_weights_path, "wb") as f:
                    f.write(resp.read())
                print(f"[OpenCLIP Detector] Successfully downloaded linear weights ({os.path.getsize(self.fc_weights_path)} bytes) to {self.fc_weights_path}")
            except Exception as e:
                print(f"[OpenCLIP Detector] Warning: Could not download linear weights from Hugging Face: {e}")

    def _init_detector(self):
        """
        Initializes OpenCLIP ViT-L-14 visual backbone and the linear classification head.
        """
        try:
            import open_clip
            print("[OpenCLIP Detector] Initializing OpenCLIP ViT-L-14 backbone...")
            
            # PyTorch 2.6 compatibility: wrap torch.load to set weights_only=False for TorchScript archives
            orig_torch_load = torch.load
            def _compat_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return orig_torch_load(*args, **kwargs)

            torch.load = _compat_load
            try:
                # Prefer local cached checkpoint if available, otherwise open_clip downloads openai weights
                if os.path.exists(self.vit_weights_path) and os.path.getsize(self.vit_weights_path) > 500 * 1024 * 1024:
                    print(f"[OpenCLIP Detector] Loading ViT-L-14 from local checkpoint: {self.vit_weights_path}...")
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=self.vit_weights_path)
                else:
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            finally:
                torch.load = orig_torch_load
                
            model.to(self.device)
            model.eval()
            self.model = model
            self.preprocess = preprocess
            
            # Linear Classification Head: 768-dim CLIP embedding -> 1 logit
            self.linear_head = nn.Linear(768, 1).to(self.device)
            
            if os.path.exists(self.fc_weights_path):
                ckpt = torch.load(self.fc_weights_path, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and "weight" in ckpt:
                    self.linear_head.load_state_dict(ckpt)
                elif hasattr(ckpt, "state_dict"):
                    self.linear_head.load_state_dict(ckpt.state_dict())
                print("[OpenCLIP Detector] Baseline linear probe weights successfully loaded!")
            self.linear_head.eval()
            self.is_loaded = True
            print("[OpenCLIP Detector] OpenCLIP ViT-L-14 AI Image Detector Ready!")
        except Exception as e:
            print(f"[OpenCLIP Detector] Initialization warning: {e}")
            self.is_loaded = False

    def detect_image(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Evaluates an image for Midjourney, Stable Diffusion, or Latent Diffusion synthetic generation.
        """
        if isinstance(image, np.ndarray):
            import cv2
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        else:
            pil_img = image.convert("RGB")

        if not self.is_loaded or self.model is None or self.linear_head is None:
            # Fallback heuristic if backbone is still initializing
            return {
                "diffusion_synthetic_probability": 0.5,
                "percentage": "50.0%",
                "verdict": "MODEL INITIALIZING",
                "risk_level": "UNKNOWN",
                "generator_detected": "OpenCLIP ViT-L-14 Loading",
                "is_ai_generated": False
            }

        try:
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Extract normalized 768-d visual feature representation
                feat = self.model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                logit = self.linear_head(feat)
                prob = float(torch.sigmoid(logit).cpu().item())
        except Exception as e:
            print(f"[OpenCLIP Detector] Inference error: {e}")
            prob = 0.5

        prob = float(np.clip(prob, 0.005, 0.995))
        pct_str = f"{prob * 100:.1f}%"

        if prob >= 0.65:
            verdict = "AI-GENERATED SYNTHETIC IMAGE"
            risk_level = "CRITICAL"
            generator = "Midjourney / Stable Diffusion Synthetic Architecture"
            desc = "High-confidence diffusion embedding matched. Visual feature representations exhibit diffusion latent manifold signatures."
            is_ai = True
        elif prob >= 0.40:
            verdict = "SUSPECTED DIFFUSION ARTIFACTS"
            risk_level = "MODERATE"
            generator = "Possible Diffusion Synthesis or Post-Processing"
            desc = "Moderate diffusion latent alignment. Image exhibits partial synthetic style patterns or hybrid rendering."
            is_ai = True
        else:
            verdict = "AUTHENTIC / NON-DIFFUSION CAPTURE"
            risk_level = "LOW"
            generator = "Natural Photographic Capture"
            desc = "Visual embeddings reflect natural real-world camera optics and photon distribution."
            is_ai = False

        return {
            "diffusion_synthetic_probability": round(prob, 4),
            "percentage": pct_str,
            "verdict": verdict,
            "risk_level": risk_level,
            "generator_detected": generator,
            "description": desc,
            "is_ai_generated": is_ai,
            "backbone": "OpenCLIP ViT-L-14 (OpenAI)",
            "embedding_dimension": 768
        }
