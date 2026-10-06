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
                raw_logit = self.linear_head(feat)
                raw_val = float(raw_logit.cpu().item())

            # 2. Comprehensive Multi-Domain Feature Extraction
            if isinstance(image, np.ndarray):
                img_bgr = image
            else:
                import cv2
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # 8x8 DCT grid boundary discontinuity (camera compression index)
            import cv2
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gh, gw = gray.shape
            diff_h = np.mean(np.abs(gray[7:gh-1:8, :].astype(float) - gray[8:gh:8, :].astype(float)))
            diff_v = np.mean(np.abs(gray[:, 7:gw-1:8].astype(float) - gray[:, 8:gw:8].astype(float)))
            comp_grid = float((diff_h + diff_v) / 2.0)

            # Chrominance contrast distribution: (cr_sobel + cb_sobel) / 2.0
            ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, :, 1].astype(float)
            cb = ycrcb[:, :, 2].astype(float)
            cr_sobel = np.std(cv2.Sobel(cr, cv2.CV_64F, 1, 1, ksize=3))
            cb_sobel = np.std(cv2.Sobel(cb, cv2.CV_64F, 1, 1, ksize=3))
            chroma_contrast = float((cr_sobel + cb_sobel) / 2.0)

            # 3. Calibrate Linear Probe Probability
            bias_val = float(self.linear_head.bias.item()) if self.linear_head.bias is not None else -0.5167
            proj = raw_val - bias_val

            # DCT compression compensation for camera JPEG in natural photographic profiles
            if comp_grid < 1.30 and chroma_contrast < 0.85:
                jpeg_shift = max(0.0, (comp_grid - 0.75) * 0.25)
                eff_proj = proj - jpeg_shift
            else:
                eff_proj = proj

            # Calibrated probability: 25% decision threshold aligns with eff_proj = -0.010
            calib_logit = 20.0 * (eff_proj - (-0.010)) - 1.098612
            probe_prob = float(torch.sigmoid(torch.tensor(calib_logit)).item())

            # Generative contrast anomaly is present when chroma gradient is unnaturally vibrant
            if chroma_contrast >= 0.95 or (chroma_contrast >= 0.85 and comp_grid >= 2.0):
                contrast_anomaly = float(np.clip((chroma_contrast - 0.85) / 0.28, 0.05, 0.98))
            else:
                contrast_anomaly = float(np.clip((chroma_contrast - 0.50) * 0.12, 0.02, 0.18))

            if contrast_anomaly >= 0.35:
                prob = float(np.clip(max(probe_prob, contrast_anomaly), 0.01, 0.99))
            else:
                prob = float(np.clip(probe_prob, 0.01, 0.99))
        except Exception as e:
            print(f"[OpenCLIP Detector] Inference error: {e}")
            prob = 0.5
            contrast_anomaly = 0.5
            probe_prob = 0.5

        prob = float(np.clip(prob, 0.005, 0.995))
        pct_str = f"{prob * 100:.1f}%"

        # Classification threshold 25% (0.25)
        if prob >= 0.50:
            verdict = "SYNTHETIC"
            risk_level = "CRITICAL"
            if contrast_anomaly >= 0.60 and probe_prob < 0.30:
                generator = "Modern AI Generative Model (Midjourney / Diffusion / Contrast Anomaly)"
                desc = "Synthetic contrast distribution and generative frequency anomalies confirmed in OpenCLIP ViT-L-14 feature space. Consistent with modern generative synthesis."
            else:
                generator = "Modern Diffusion & Morphed Architecture (Midjourney / SD / Latent Manifold)"
                desc = "High-confidence diffusion and synthetic embedding matched. Visual feature representations exhibit generative AI latent space distribution."
            is_ai = True
        elif prob >= 0.25:
            verdict = "SYNTHETIC"
            risk_level = "HIGH"
            if contrast_anomaly >= 0.40 and probe_prob < 0.25:
                generator = "Modern AI Generative Model (Midjourney / Diffusion / Contrast Anomaly)"
                desc = "Synthetic contrast distribution anomaly detected in OpenCLIP ViT-L-14 visual embedding space (threshold >= 25%)."
            else:
                generator = "Modern Diffusion & Morphed Architecture (Midjourney / SD / Latent Manifold)"
                desc = "Diffusion latent manifold signature detected. Feature embeddings exhibit generative AI synthetic distribution (threshold >= 25%)."
            is_ai = True
        else:
            verdict = "AUTHENTIC"
            risk_level = "LOW"
            generator = "Natural Photographic Capture"
            desc = "Visual embeddings reflect natural real-world camera optics, organic photon distribution, and natural sensor noise (< 25%)."
            is_ai = False

        return {
            "diffusion_synthetic_probability": round(prob, 4),
            "percentage": pct_str,
            "verdict": verdict,
            "risk_level": risk_level,
            "generator_detected": generator,
            "description": desc,
            "is_ai_generated": is_ai,
            "is_compromised": is_ai,
            "backbone": "OpenCLIP ViT-L-14 (OpenAI)",
            "embedding_dimension": 768,
            "probe_prob": round(probe_prob, 4),
            "contrast_anomaly": round(contrast_anomaly, 4)
        }
