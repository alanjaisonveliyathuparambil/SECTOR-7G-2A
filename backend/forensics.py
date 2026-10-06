import os
from typing import Dict, Any, List
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from models.xception import Xception
from models.selim_dfdc import SelimDFDCClassifier
from models.wang_cnndetect import WangCNNDetector
from models.clip_diffusion_detector import OpenCLIPDiffusionDetector

class ForensicEngine:
    def __init__(self, weights_dir: str = "weights", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.weights_dir = weights_dir
        self.models_loaded = {}
        
        # Transforms
        # Xception: 299x299, normalized with mean 0.5, std 0.5
        self.xception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Selim DFDC: 380x380, normalized ImageNet
        self.selim_transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Sheng-Yu Wang CNNDetection: 256x256 -> 224x224 center crop, normalized ImageNet
        self.wang_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # OpenCLIP ViT-L-14 Midjourney & Stable Diffusion Detector
        self.clip_detector = None
        self._load_models()

    def _load_models(self):
        # 1. Load Xception (FaceForensics++ / Deepfake Classifier)
        xception_weight_path = os.path.join(self.weights_dir, "best_xception.pth")
        if os.path.exists(xception_weight_path):
            try:
                print(f"[ForensicEngine] Loading FaceForensics++ Xception from {xception_weight_path}...")
                model_x = Xception(num_classes=2)
                sd = torch.load(xception_weight_path, map_location=self.device, weights_only=False)
                model_x.load_state_dict(sd)
                model_x.to(self.device)
                model_x.eval()
                self.models_loaded["faceforensics_xception"] = model_x
                print("[ForensicEngine] FaceForensics++ Xception loaded successfully!")
            except Exception as e:
                print(f"[ForensicEngine] Warning: Could not load Xception: {e}")

        # 2. Load Selim DFDC EfficientNet-B7
        selim_weight_path = os.path.join(self.weights_dir, "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36")
        if os.path.exists(selim_weight_path):
            try:
                print(f"[ForensicEngine] Loading Selim DFDC model from {selim_weight_path}...")
                model_s = SelimDFDCClassifier.load_from_checkpoint(selim_weight_path, device=self.device)
                self.models_loaded["selim_dfdc"] = model_s
                print("[ForensicEngine] Selim DFDC model loaded successfully!")
            except Exception as e:
                print(f"[ForensicEngine] Warning: Could not load Selim DFDC: {e}")

        # 3. Load Sheng-Yu Wang CNNDetection ResNet-50
        wang_weight_path = os.path.join(self.weights_dir, "blur_jpg_prob0.5.pth")
        if os.path.exists(wang_weight_path):
            try:
                print(f"[ForensicEngine] Loading Sheng-Yu Wang CNNDetection from {wang_weight_path}...")
                model_w = WangCNNDetector.load_from_checkpoint(wang_weight_path, device=self.device)
                self.models_loaded["wang_cnndetect"] = model_w
                print("[ForensicEngine] Sheng-Yu Wang CNNDetection loaded successfully!")
            except Exception as e:
                print(f"[ForensicEngine] Warning: Could not load Sheng-Yu Wang CNNDetection: {e}")

        # 4. Initialize OpenCLIP ViT-L-14 Detector (Diffusion / Midjourney)
        try:
            print("[ForensicEngine] Initializing OpenCLIP ViT-L-14 Diffusion Detector...")
            self.clip_detector = OpenCLIPDiffusionDetector(weights_dir=self.weights_dir, device=self.device)
            if self.clip_detector.is_loaded:
                self.models_loaded["clip_vit_l14"] = self.clip_detector
                print("[ForensicEngine] OpenCLIP ViT-L-14 Diffusion Detector loaded successfully!")
        except Exception as e:
            print(f"[ForensicEngine] Warning: OpenCLIP ViT-L-14 initialization deferred: {e}")

    def evaluate_diffusion_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates image for Midjourney / Stable Diffusion generation using OpenCLIP ViT-L-14.
        """
        if self.clip_detector is None or not self.clip_detector.is_loaded:
            # Attempt re-init if weights were just downloaded
            try:
                self.clip_detector = OpenCLIPDiffusionDetector(weights_dir=self.weights_dir, device=self.device)
                if self.clip_detector.is_loaded:
                    self.models_loaded["clip_vit_l14"] = self.clip_detector
            except Exception:
                pass

        if self.clip_detector and self.clip_detector.is_loaded:
            return self.clip_detector.detect_image(image_bgr)
        else:
            return {
                "diffusion_synthetic_probability": 0.5,
                "percentage": "50.0%",
                "verdict": "MODEL INITIALIZING",
                "risk_level": "UNKNOWN",
                "generator_detected": "OpenCLIP ViT-L-14 Loading",
                "is_ai_generated": False
            }

    def evaluate_gan_footprint(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates static image for universal GAN structural artifacts using
        Sheng-Yu Wang's CNNDetection model combined with 2D FFT spectral decomposition.
        """
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        model_prob = 0.5
        if "wang_cnndetect" in self.models_loaded:
            try:
                w, h = pil_img.size
                tensors = [self.wang_transform(pil_img).unsqueeze(0)]
                
                if min(w, h) >= 300:
                    crop_tl = pil_img.crop((0, 0, int(w*0.6), int(h*0.6)))
                    crop_br = pil_img.crop((int(w*0.4), int(h*0.4), w, h))
                    tensors.append(self.wang_transform(crop_tl).unsqueeze(0))
                    tensors.append(self.wang_transform(crop_br).unsqueeze(0))
                
                batch = torch.cat(tensors, dim=0).to(self.device)
                with torch.no_grad():
                    probs, _ = self.models_loaded["wang_cnndetect"](batch)
                    model_prob = float(probs.mean().cpu().item())
            except Exception as e:
                print(f"[ForensicEngine] Error during Wang CNNDetection evaluation: {e}")
                model_prob = 0.5

        spectral_score = self.analyze_frequency_domain(image_bgr)
        raw_score = 0.80 * model_prob + 0.20 * spectral_score
        footprint_score = float(np.clip(raw_score, 0.01, 0.99))
        
        if footprint_score >= 0.70:
            footprint_verdict = "EVIDENT GAN/SYNTHETIC FOOTPRINT"
            footprint_risk = "HIGH"
            footprint_desc = "Distinct CNN generator artifacts and checkerboard lattice signatures detected. Image exhibits structural markers of ProGAN/StyleGAN/Diffusion synthesis."
        elif footprint_score >= 0.40:
            footprint_verdict = "SUSPECTED GAN ARTIFACTS"
            footprint_risk = "MODERATE"
            footprint_desc = "Mild structural anomalies present. Some upsampling or resampling noise patterns consistent with post-processing or synthetic enhancement."
        else:
            footprint_verdict = "ORGANIC NATURAL CAPTURE"
            footprint_risk = "LOW"
            footprint_desc = "Organic sensor noise profile and natural frequency decay. No dominant CNN architectural footprint detected."

        return {
            "gan_synthetic_footprint_score": round(footprint_score, 4),
            "percentage": f"{footprint_score * 100:.1f}%",
            "verdict": footprint_verdict,
            "risk_level": footprint_risk,
            "description": footprint_desc,
            "cnn_generator_probability": round(model_prob, 4),
            "spectral_lattice_anomaly": round(spectral_score, 4)
        }

    def analyze_frequency_domain(self, crop_bgr: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
            
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            r = 30
            mask[crow - r:crow + r, ccol - r:ccol + r] = 0
            
            high_freq_mag = magnitude_spectrum * mask
            hf_std = np.std(high_freq_mag[mask == 1])
            
            score = float(np.clip((hf_std - 18.0) / 25.0, 0.05, 0.95))
            return round(score, 3)
        except Exception:
            return 0.25

    def analyze_spatial_laplacian(self, crop_bgr: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            if variance < 90:
                score = 0.75
            elif variance > 650:
                score = 0.70
            else:
                score = 0.15
            return round(score, 3)
        except Exception:
            return 0.20

    def analyze_chrominance_mismatch(self, crop_bgr: np.ndarray) -> float:
        try:
            ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
            h, w = ycrcb.shape[:2]
            
            inner = ycrcb[h//4:3*h//4, w//4:3*w//4]
            mean_cr_inner = np.mean(inner[:, :, 1])
            mean_cb_inner = np.mean(inner[:, :, 2])
            
            mean_cr_outer = (np.sum(ycrcb[:, :, 1]) - np.sum(inner[:, :, 1])) / max((h*w - inner.shape[0]*inner.shape[1]), 1)
            mean_cb_outer = (np.sum(ycrcb[:, :, 2]) - np.sum(inner[:, :, 2])) / max((h*w - inner.shape[0]*inner.shape[1]), 1)
            
            diff = np.sqrt((mean_cr_inner - mean_cr_outer)**2 + (mean_cb_inner - mean_cb_outer)**2)
            score = float(np.clip(diff / 22.0, 0.05, 0.90))
            return round(score, 3)
        except Exception:
            return 0.22

    def predict_crop(self, crop_bgr: np.ndarray, model_name: str = "ensemble") -> Dict[str, Any]:
        """
        Runs deep neural network inference on a cropped region.
        """
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        scores = {}
        
        # 1. FaceForensics++ Xception Inference
        if "faceforensics_xception" in self.models_loaded:
            try:
                x_tensor = self.xception_transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.models_loaded["faceforensics_xception"](x_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    scores["xception_fake_prob"] = round(float(probs[0]), 4)
            except Exception as e:
                scores["xception_fake_prob"] = 0.5
        else:
            scores["xception_fake_prob"] = 0.5

        # 2. Selim DFDC EfficientNet-B7 Inference
        if "selim_dfdc" in self.models_loaded:
            try:
                s_tensor = self.selim_transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob, _ = self.models_loaded["selim_dfdc"](s_tensor)
                    scores["selim_fake_prob"] = round(float(prob.cpu().numpy()[0][0]), 4)
            except Exception as e:
                scores["selim_fake_prob"] = 0.5
        else:
            scores["selim_fake_prob"] = 0.5

        # 3. Sheng-Yu Wang CNNDetection Inference
        if "wang_cnndetect" in self.models_loaded:
            try:
                w_tensor = self.wang_transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    w_prob, _ = self.models_loaded["wang_cnndetect"](w_tensor)
                    scores["wang_gan_prob"] = round(float(w_prob.cpu().numpy()[0][0]), 4)
            except Exception as e:
                scores["wang_gan_prob"] = 0.5
        else:
            scores["wang_gan_prob"] = 0.5

        # 4. OpenCLIP ViT-L-14 Diffusion Inference
        if self.clip_detector and self.clip_detector.is_loaded:
            try:
                diff_res = self.clip_detector.detect_image(pil_img)
                scores["clip_diffusion_prob"] = diff_res["diffusion_synthetic_probability"]
            except Exception:
                scores["clip_diffusion_prob"] = 0.5
        else:
            scores["clip_diffusion_prob"] = 0.5

        # Signals
        freq_score = self.analyze_frequency_domain(crop_bgr)
        spatial_score = self.analyze_spatial_laplacian(crop_bgr)
        color_score = self.analyze_chrominance_mismatch(crop_bgr)

        # Compute Final Probability based on selected model
        if model_name == "selim_dfdc":
            base_fake = scores.get("selim_fake_prob", 0.5)
            final_fake = 0.85 * base_fake + 0.15 * (0.5 * freq_score + 0.5 * spatial_score)
        elif model_name == "faceforensics_xception":
            base_fake = scores.get("xception_fake_prob", 0.5)
            final_fake = 0.85 * base_fake + 0.15 * (0.5 * freq_score + 0.5 * spatial_score)
        elif model_name == "wang_cnndetect":
            base_fake = scores.get("wang_gan_prob", 0.5)
            final_fake = 0.85 * base_fake + 0.15 * (0.5 * freq_score + 0.5 * spatial_score)
        elif model_name == "clip_diffusion":
            base_fake = scores.get("clip_diffusion_prob", 0.5)
            final_fake = 0.85 * base_fake + 0.15 * freq_score
        else:
            # Multi-Model Ensemble Consensus
            p_s = scores.get("selim_fake_prob", 0.5)
            p_x = scores.get("xception_fake_prob", 0.5)
            p_w = scores.get("wang_gan_prob", 0.5)
            p_c = scores.get("clip_diffusion_prob", 0.5)
            p_forensic = (freq_score * 0.4 + spatial_score * 0.3 + color_score * 0.3)
            final_fake = 0.30 * p_s + 0.30 * p_x + 0.20 * p_w + 0.15 * p_c + 0.05 * p_forensic

        final_fake = float(np.clip(final_fake, 0.01, 0.99))
        real_prob = 1.0 - final_fake

        if final_fake >= 0.70:
            verdict = "DEEPFAKE DETECTED"
            risk_level = "CRITICAL"
        elif final_fake >= 0.45:
            verdict = "SUSPICIOUS MANIPULATION"
            risk_level = "MODERATE"
        else:
            verdict = "VERIFIED AUTHENTIC"
            risk_level = "LOW"

        return {
            "fake_probability": round(final_fake, 4),
            "real_probability": round(real_prob, 4),
            "verdict": verdict,
            "risk_level": risk_level,
            "model_scores": scores,
            "signals": {
                "spectral_anomaly": freq_score,
                "spatial_discontinuity": spatial_score,
                "chrominance_mismatch": color_score,
                "compression_artifact_index": round(float(np.clip(freq_score * 0.7 + color_score * 0.3, 0.1, 0.9)), 3)
            }
        }
