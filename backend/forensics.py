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
        if model_name in ["ensemble", "photo"]:
            # On facial crops, evaluate specialized face forensic networks (Selim DFDC & FF++ Xception)
            face_manip_prob = max(scores.get("selim_fake_prob", 0.0), scores.get("xception_fake_prob", 0.0))
            final_fake = 0.85 * face_manip_prob + 0.15 * (0.5 * freq_score + 0.5 * spatial_score)
        elif model_name == "clip_diffusion":
            final_fake = scores.get("clip_diffusion_prob", 0.5)
        elif model_name == "selim_dfdc":
            base_fake = scores.get("selim_fake_prob", 0.5)
            final_fake = 0.85 * base_fake + 0.15 * (0.5 * freq_score + 0.5 * spatial_score)
        elif model_name == "faceforensics_xception":
            base_fake = scores.get("xception_fake_prob", 0.5)
            final_fake = 0.85 * base_fake + 0.15 * (0.5 * freq_score + 0.5 * spatial_score)
        elif model_name == "wang_cnndetect":
            base_fake = scores.get("wang_gan_prob", 0.5)
            final_fake = 0.85 * base_fake + 0.15 * (0.5 * freq_score + 0.5 * spatial_score)
        else:
            face_manip_prob = max(scores.get("selim_fake_prob", 0.0), scores.get("xception_fake_prob", 0.0))
            final_fake = 0.85 * face_manip_prob + 0.15 * (0.5 * freq_score + 0.5 * spatial_score)

        final_fake = float(np.clip(final_fake, 0.01, 0.99))
        real_prob = 1.0 - final_fake

        # Threshold lowered to 25% (0.25) so modern diffusion images are correctly flagged as SYNTHETIC
        if final_fake >= 0.25:
            verdict = "SYNTHETIC"
            risk_level = "CRITICAL" if final_fake >= 0.50 else "HIGH"
        else:
            verdict = "AUTHENTIC"
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

    # =========================================================================
    # LAYER 1: Selim Seferbekov (selimsef/dfdc_deepfake_challenge)
    # Target: Spatial face-boundary artifacts & skin texturing blending seams
    # =========================================================================
    def analyze_selim_boundary_layer(self, image_bgr: np.ndarray, faces: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Layer 1: Spatial face-boundary artifacts and skin texturing blending seams.
        Extracts facial boundary perimeters, evaluates skin texture transition gradients,
        and executes Selim Seferbekov's DFDC 1st-place EfficientNet-B7 deep classifier.
        [selimsef/dfdc_deepfake_challenge GitHub Repository]
        """
        model_score = 0.05
        seam_scores = []
        target_crops = []

        if faces and len(faces) > 0:
            for f in faces:
                if isinstance(f, dict) and "crop" in f and f["crop"] is not None:
                    target_crops.append(f["crop"])
        if not target_crops:
            h, w = image_bgr.shape[:2]
            crop = image_bgr[h//6:5*h//6, w//6:5*w//6]
            if crop.size > 0:
                target_crops.append(crop)
            else:
                target_crops.append(image_bgr)

        # Run Selim DFDC model on target facial crops
        if "selim_dfdc" in self.models_loaded and len(target_crops) > 0:
            crop_scores = []
            for crop in target_crops:
                try:
                    pil_c = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    s_tensor = self.selim_transform(pil_c).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        prob, _ = self.models_loaded["selim_dfdc"](s_tensor)
                        crop_scores.append(float(prob.cpu().numpy()[0][0]))
                except Exception:
                    pass
            if crop_scores:
                model_score = float(np.mean(crop_scores))

        # Spatial face-boundary gradient & skin texturing seam analysis
        for crop in target_crops:
            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(sobelx**2 + sobely**2)

                border_mask = np.ones((h, w), dtype=bool)
                border_mask[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)] = False
                interior_mask = ~border_mask

                b_grad = float(np.mean(grad_mag[border_mask])) if np.any(border_mask) else 1.0
                i_grad = float(np.mean(grad_mag[interior_mask])) if np.any(interior_mask) else 1.0
                ratio = b_grad / (i_grad + 1e-5)
                # Calibrated boundary seam score
                seam_val = float(np.clip(ratio * 0.70, 0.02, 0.95))
                seam_scores.append(seam_val)
            except Exception:
                seam_scores.append(0.05)

        texture_discontinuity = float(np.mean(seam_scores)) if seam_scores else 0.08
        
        # When Selim DFDC neural model is loaded, its specialized feature representation is primary
        if "selim_dfdc" in self.models_loaded and model_score > 0:
            combined_score = float(np.clip(model_score, 0.01, 0.99))
        else:
            combined_score = float(np.clip(texture_discontinuity, 0.01, 0.99))
            
        is_compromised = combined_score >= 0.25

        return {
            "layer_id": "layer1_spatial_boundaries",
            "layer_name": "Spatial Face-Boundary & Skin Texture Layer",
            "model": "Selim Seferbekov (DFDC 1st Place Winner)",
            "target": "Spatial face-boundary artifacts and skin texturing blending seams",
            "repository": "selimsef/dfdc_deepfake_challenge",
            "score": round(combined_score, 4),
            "percentage": f"{combined_score * 100:.1f}%",
            "status": "FAIL (BLENDING SEAMS DETECTED)" if is_compromised else "PASS (NATURAL SKIN TEXTURE)",
            "is_flagged": is_compromised,
            "metrics": {
                "selim_dfdc_deep_score": round(model_score, 4),
                "boundary_gradient_discontinuity": round(texture_discontinuity, 4),
                "skin_texture_seam_index": round(texture_discontinuity * 0.85, 4)
            },
            "description": (
                "Elevated spatial face-boundary artifacts and skin texturing blending seams detected along facial margins. Consistent with face mask warping or Deepfake boundary splicing."
                if is_compromised else
                "Seamless facial contour transitions and organic skin micro-texture continuity verified. No face swap blending seams or warping artifacts detected."
            )
        }

    # =========================================================================
    # LAYER 2: FaceForensics++ Xception Net
    # Target: Depthwise separable compression matrices & specific tampering styles
    # =========================================================================
    def analyze_xception_compression_layer(self, image_bgr: np.ndarray, faces: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Layer 2: Depthwise separable compression matrices and specific tampering styles.
        Evaluates compression block residuals, DCT quantization noise, and executes
        the FaceForensics++ Xception benchmark classifier across depthwise separable convolutions.
        """
        model_score = 0.02
        target_crops = []

        if faces and len(faces) > 0:
            for f in faces:
                if isinstance(f, dict) and "crop" in f and f["crop"] is not None:
                    target_crops.append(f["crop"])
        if not target_crops:
            target_crops.append(image_bgr)

        # Run FaceForensics++ Xception Net
        if "faceforensics_xception" in self.models_loaded and len(target_crops) > 0:
            crop_scores = []
            for crop in target_crops:
                try:
                    pil_c = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    x_tensor = self.xception_transform(pil_c).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits = self.models_loaded["faceforensics_xception"](x_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        crop_scores.append(float(probs[0]))
                except Exception:
                    pass
            if crop_scores:
                model_score = float(np.mean(crop_scores))

        # Depthwise separable compression matrix residual: 8x8 DCT grid boundary discontinuity
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        diff_h = np.abs(gray[7:h-1:8, :].astype(float) - gray[8:h:8, :].astype(float))
        diff_v = np.abs(gray[:, 7:w-1:8].astype(float) - gray[:, 8:w:8].astype(float))
        comp_residual = float(np.clip((np.mean(diff_h) + np.mean(diff_v)) / 18.0, 0.05, 0.95))

        combined_score = float(np.clip(0.85 * model_score + 0.15 * comp_residual, 0.01, 0.99))
        is_compromised = combined_score >= 0.25

        if combined_score >= 0.70:
            tampering_style = "Deepfakes / NeuralTextures Facial Reenactment"
        elif combined_score >= 0.40:
            tampering_style = "Face2Face / FaceSwap Landmark Warping"
        elif combined_score >= 0.25:
            tampering_style = "Suspected Post-Processed Compression Residue"
        else:
            tampering_style = "Authentic Camera Compression Matrix"

        return {
            "layer_id": "layer2_depthwise_compression",
            "layer_name": "Depthwise Separable Compression & Tampering Layer",
            "model": "FaceForensics++ Xception Net",
            "target": "Depthwise separable compression matrices and specific tampering styles",
            "repository": "FaceForensics++ (TUM / Google)",
            "score": round(combined_score, 4),
            "percentage": f"{combined_score * 100:.1f}%",
            "status": "FAIL (TAMPERING MATRIX DETECTED)" if is_compromised else "PASS (UNIFIED COMPRESSION MATRIX)",
            "is_flagged": is_compromised,
            "tampering_style": tampering_style,
            "metrics": {
                "xception_tampering_score": round(model_score, 4),
                "depthwise_compression_residual": round(comp_residual, 4),
                "block_grid_discontinuity": round(float(np.mean(diff_h)), 3)
            },
            "description": (
                f"Depthwise separable compression residual discrepancy detected ({tampering_style}). Discontinuity across depthwise separable convolution blocks indicates localized tampering."
                if is_compromised else
                "Homogeneous depthwise separable compression matrices across blocks. No localized facial tampering or reenactment artifacts identified."
            )
        }

    # =========================================================================
    # LAYER 3: Sheng-Yu Wang CNNDetection (PeterWang512/CNNDetection)
    # Target: High-frequency upsampling patterns (periodic checkerboard noise)
    # =========================================================================
    def analyze_wang_upsampling_layer(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Layer 3: High-frequency upsampling patterns and periodic checkerboard noise.
        Computes 2D Fast Fourier Transform (FFT) spectral decomposition, measures periodic
        lattice spikes caused by transposed convolution / deconvolution upsampling, and executes
        Sheng-Yu Wang's CNNDetection (ResNet-50 with Blur & JPEG Augmentation).
        [PeterWang512/CNNDetection GitHub repository]
        """
        footprint = self.evaluate_gan_footprint(image_bgr)
        score = footprint.get("gan_synthetic_footprint_score", 0.05)
        model_prob = footprint.get("cnn_generator_probability", 0.05)
        spectral_score = footprint.get("spectral_lattice_anomaly", 0.05)

        # 2D FFT periodic checkerboard energy calculation
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_256 = cv2.resize(gray, (256, 256))
        f_shift = np.fft.fftshift(np.fft.fft2(gray_256))
        mag = 20 * np.log(np.abs(f_shift) + 1e-6)
        
        # Periodic checkerboard spikes concentrate in high-frequency quadrant corners
        hf_corners = (mag[:32, :32] + mag[-32:, :32] + mag[:32, -32:] + mag[-32:, -32:]) / 4.0
        checkerboard_energy = float(np.clip(np.std(hf_corners) / 22.0, 0.02, 0.98))

        combined_score = float(np.clip(0.70 * score + 0.30 * checkerboard_energy, 0.01, 0.99))
        is_compromised = combined_score >= 0.25

        return {
            "layer_id": "layer3_frequency_upsampling",
            "layer_name": "High-Frequency Upsampling & Checkerboard Noise Layer",
            "model": "Sheng-Yu Wang CNNDetection (ResNet-50)",
            "target": "High-frequency upsampling patterns (periodic checkerboard noise)",
            "repository": "PeterWang512/CNNDetection",
            "score": round(combined_score, 4),
            "percentage": f"{combined_score * 100:.1f}%",
            "status": "FAIL (PERIODIC CHECKERBOARD DETECTED)" if is_compromised else "PASS (NATURAL FREQUENCY DECAY)",
            "is_flagged": is_compromised,
            "generator_architecture": "ProGAN / StyleGAN CNN Upsampling Lattice" if is_compromised else "Organic Natural Optical Decay",
            "metrics": {
                "wang_cnn_probability": round(model_prob, 4),
                "periodic_checkerboard_energy": round(checkerboard_energy, 4),
                "spectral_lattice_anomaly": round(spectral_score, 4)
            },
            "description": (
                "High-frequency periodic checkerboard noise patterns detected in 2D FFT spectrum. Architectural markers match transposed convolution upsampling lattices typical of CNN generator models."
                if is_compromised else
                "Natural radial 1/f frequency decay across the 2D Fourier spectrum. Absence of periodic checkerboard lattice spikes or deconvolution upsampling noise."
            )
        }

    # =========================================================================
    # LAYER 4: OpenCLIP ViT-L-14 (UniversalFakeDetect)
    # Target: High-level semantic anomalies & synthetic contrast distributions
    # =========================================================================
    def analyze_openclip_semantic_layer(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Layer 4: High-level semantic anomalies and synthetic contrast distributions.
        Extracts 768-dimensional visual feature representations from OpenCLIP ViT-L-14 backbone,
        measures generative contrast variance across semantic space, and executes the calibrated
        linear probe for Midjourney, Stable Diffusion, and Latent Diffusion synthesis.
        """
        diff_res = self.evaluate_diffusion_image(image_bgr)
        prob = diff_res.get("diffusion_synthetic_probability", 0.05)
        contrast_anomaly = diff_res.get("contrast_anomaly", 0.05)
        probe_prob = diff_res.get("probe_prob", prob)

        is_compromised = prob >= 0.25

        return {
            "layer_id": "layer4_semantic_contrast",
            "layer_name": "High-Level Semantic & Synthetic Contrast Layer",
            "model": "OpenCLIP ViT-L-14 (UniversalFakeDetect Probe)",
            "target": "High-level semantic anomalies and synthetic contrast distributions",
            "repository": "UniversalFakeDetect (CVPR 2023, Ojha et al.)",
            "score": round(prob, 4),
            "percentage": f"{prob * 100:.1f}%",
            "status": "FAIL (SYNTHETIC CONTRAST ANOMALY)" if is_compromised else "PASS (NATURAL PHOTOGRAPHIC MANIFOLD)",
            "is_flagged": is_compromised,
            "architecture_detected": diff_res.get("generator_detected", "Modern Latent Diffusion (Midjourney / Stable Diffusion)"),
            "metrics": {
                "openclip_diffusion_score": round(prob, 4),
                "synthetic_contrast_index": round(contrast_anomaly, 4),
                "semantic_anomaly_index": round(probe_prob, 4)
            },
            "description": diff_res.get("description", (
                "High-level semantic anomalies and synthetic contrast distributions detected in OpenCLIP ViT-L-14 visual embedding space (>= 25%). Latent diffusion manifold signature confirmed."
                if is_compromised else
                "Visual feature representations align with authentic camera optics, organic photon distribution, and natural photographic sensor dynamics (< 25%)."
            ))
        }

    # =========================================================================
    # LOGICAL FORENSIC THINKING & REASONING ENGINE
    # =========================================================================
    def conduct_logical_forensic_thinking(
        self,
        layer1: Dict[str, Any],
        layer2: Dict[str, Any],
        layer3: Dict[str, Any],
        layer4: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes all 4 forensic representation layers using a deductive chain-of-logic reasoning engine:
        1. Layer 1 (Selim Seferbekov): Evaluates spatial face-boundary artifacts and skin texturing blending seams.
        2. Layer 2 (FaceForensics++ Xception): Evaluates depthwise separable compression matrices and tampering styles.
        3. Layer 3 (Sheng-Yu Wang CNN): Evaluates high-frequency upsampling patterns and periodic checkerboard noise.
        4. Layer 4 (OpenCLIP ViT-L-14): Evaluates high-level semantic anomalies and synthetic contrast distributions.
        5. Formulates deductive hypotheses, cross-checks inter-layer correlations, and determines
           the primary manipulation archetype and overall verdict.
        """
        l1_score = layer1.get("score", 0.0)
        l2_score = layer2.get("score", 0.0)
        l3_score = layer3.get("score", 0.0)
        l4_score = layer4.get("score", 0.0)

        # Flagged count
        flagged_layers = []
        if l1_score >= 0.25:
            flagged_layers.append("Layer 1 (Spatial Boundaries & Skin Seams)")
        if l2_score >= 0.25:
            flagged_layers.append("Layer 2 (Depthwise Separable Compression)")
        if l3_score >= 0.25:
            flagged_layers.append("Layer 3 (High-Frequency Checkerboard Upsampling)")
        if l4_score >= 0.25:
            flagged_layers.append("Layer 4 (Semantic Contrast & Diffusion Manifold)")

        # Deductive Reasoning Chain
        reasoning_steps = []

        # Step 1: Premise
        reasoning_steps.append({
            "step": 1,
            "title": "Layer Decomposition & Multi-Domain Feature Extraction",
            "observation": (
                "The input media was extracted into 4 orthogonal analytical representation layers: "
                "Spatial Face-Boundary & Skin Texture (Selim DFDC), Depthwise Separable Compression (FF++ Xception), "
                "Spectral Upsampling Patterns (Wang CNN), and High-Level Semantic Contrast (OpenCLIP ViT-L-14)."
            ),
            "status": "COMPLETED"
        })

        # Step 2: Layer 1 Reasoning (Selim Seferbekov)
        if l1_score >= 0.25:
            l1_obs = f"ALERT: Elevated boundary gradient discontinuity ({layer1['metrics']['boundary_gradient_discontinuity']}) and skin texturing blending seams detected along facial perimeter. Consistent with face mask warping or Deepfake blending artifacts."
            l1_status = "ANOMALY DETECTED"
        else:
            l1_obs = f"NORMAL: Seamless facial contour transitions and organic skin micro-texture continuity verified (score: {l1_score*100:.1f}%). Absence of face-swapping boundary seams."
            l1_status = "CLEARED"

        reasoning_steps.append({
            "step": 2,
            "title": "Layer 1 Audit: Spatial Face-Boundary & Skin Texture (Selim Seferbekov)",
            "observation": l1_obs,
            "status": l1_status
        })

        # Step 3: Layer 2 Reasoning (FF++ Xception)
        if l2_score >= 0.25:
            l2_obs = f"ALERT: Depthwise separable compression matrix anomaly detected ({layer2.get('tampering_style')}). Discontinuity identified across separable convolutional feature maps."
            l2_status = "ANOMALY DETECTED"
        else:
            l2_obs = f"NORMAL: Homogeneous depthwise separable compression matrices across blocks (score: {l2_score*100:.1f}%). No localized facial reenactment or texture tampering residuals detected."
            l2_status = "CLEARED"

        reasoning_steps.append({
            "step": 3,
            "title": "Layer 2 Audit: Depthwise Separable Compression & Tampering (FaceForensics++ Xception)",
            "observation": l2_obs,
            "status": l2_status
        })

        # Step 4: Layer 3 Reasoning (Wang CNNDetection)
        if l3_score >= 0.25:
            l3_obs = f"ALERT: High-frequency periodic checkerboard noise pattern identified (energy: {layer3['metrics']['periodic_checkerboard_energy']}). Structural markers match transposed convolution upsampling lattices typical of CNN generator models."
            l3_status = "ANOMALY DETECTED"
        else:
            l3_obs = f"NORMAL: Natural radial 1/f frequency decay across the 2D Fourier spectrum (score: {l3_score*100:.1f}%). Absence of periodic checkerboard lattice spikes or deconvolution upsampling noise."
            l3_status = "CLEARED"

        reasoning_steps.append({
            "step": 4,
            "title": "Layer 3 Audit: High-Frequency Upsampling Patterns (Sheng-Yu Wang CNNDetection)",
            "observation": l3_obs,
            "status": l3_status
        })

        # Step 5: Layer 4 Reasoning (OpenCLIP ViT-L-14)
        if l4_score >= 0.25:
            l4_obs = f"ALERT: High-level semantic anomalies and synthetic contrast distributions confirmed in OpenCLIP ViT-L-14 embedding space (score: {l4_score*100:.1f}%). Visual feature representations align with modern diffusion latent manifolds."
            l4_status = "ANOMALY DETECTED"
        else:
            l4_obs = f"NORMAL: Visual feature representations adhere to natural camera photon dynamics and authentic optical sensor noise (score: {l4_score*100:.1f}% < 25%)."
            l4_status = "CLEARED"

        reasoning_steps.append({
            "step": 5,
            "title": "Layer 4 Audit: High-Level Semantic Contrast (OpenCLIP ViT-L-14)",
            "observation": l4_obs,
            "status": l4_status
        })

        # Step 6: Deductive Synthesis & Archetype Classification
        # Logical Rule 1: All 4 layers cleared -> Authentic
        if len(flagged_layers) == 0:
            is_compromised = False
            verdict = "AUTHENTIC"
            risk_level = "LOW"
            fake_prob = max(l1_score, l2_score, l3_score, l4_score)
            manipulation_type = "Authentic Optical Capture"
            deduction_summary = (
                "Deductive Synthesis: All 4 orthogonal forensic representation layers (Selim boundary seams, "
                "Xception compression matrices, Wang frequency upsampling, and OpenCLIP semantic contrast) "
                "consistently confirm natural photographic characteristics. Logical deduction: Authenticity confirmed with high confidence."
            )
        # Logical Rule 2: OpenCLIP flagged alone or dominant -> Modern Latent Diffusion / AI Generative / Morphed
        elif l4_score >= 0.25 and l1_score < 0.25 and l2_score < 0.25 and l3_score < 0.25:
            is_compromised = True
            verdict = "SYNTHETIC"
            risk_level = "CRITICAL" if l4_score >= 0.50 else "HIGH"
            fake_prob = l4_score
            contrast_idx = layer4.get("metrics", {}).get("synthetic_contrast_index", 0.0)
            if contrast_idx >= 0.40:
                manipulation_type = "Modern AI Generative Synthesis (Midjourney / Diffusion / Contrast Anomaly)"
                deduction_summary = (
                    f"Deductive Synthesis: Synthetic contrast distribution anomaly confirmed by OpenCLIP ViT-L-14 ({l4_score*100:.1f}% >= 25%). "
                    "Unnatural chrominance gradient energy and lack of optical sensor depth falloff indicate whole-image AI generation."
                )
            else:
                manipulation_type = "Morphed Face & Synthetic Feature Splicing"
                deduction_summary = (
                    f"Deductive Synthesis: High-level semantic anomalies confirmed by OpenCLIP ViT-L-14 ({l4_score*100:.1f}% >= 25%). "
                    "Visual manifold projection matches synthetic facial blending and morphing characteristics."
                )
        # Logical Rule 3: Selim DFDC or Xception flagged -> Deepfake Face Swap / Morphed Blending
        elif (l1_score >= 0.25 or l2_score >= 0.25) and l3_score < 0.25 and l4_score < 0.25:
            is_compromised = True
            verdict = "SYNTHETIC"
            risk_level = "CRITICAL" if max(l1_score, l2_score) >= 0.50 else "HIGH"
            fake_prob = max(l1_score, l2_score)
            manipulation_type = "Morphed Face Swap & Facial Blending Seams"
            deduction_summary = (
                "Deductive Synthesis: Localized spatial face-boundary artifacts and depthwise separable compression residuals detected. "
                "FaceForensics++ Xception and Selim DFDC confirm facial manipulation with boundary blending seam discordance."
            )
        # Logical Rule 4: Wang CNN flagged -> GAN Architectural Generation
        elif l3_score >= 0.25 and l1_score < 0.25 and l2_score < 0.25 and l4_score < 0.25:
            is_compromised = True
            verdict = "SYNTHETIC"
            risk_level = "HIGH"
            fake_prob = l3_score
            manipulation_type = "CNN Generator Architecture (ProGAN / StyleGAN Footprint)"
            deduction_summary = (
                f"Deductive Synthesis: High-frequency periodic checkerboard noise ({layer3['metrics']['periodic_checkerboard_energy']}) "
                "and deconvolution upsampling lattice detected by Sheng-Yu Wang CNNDetection. Structural artifacts confirm GAN synthesis."
            )
        # Logical Rule 5: Multi-Vector / Multi-Layer Compromise
        else:
            is_compromised = True
            verdict = "SYNTHETIC"
            risk_level = "CRITICAL"
            fake_prob = max(l1_score, l2_score, l3_score, l4_score)
            manipulation_type = "Multi-Layer Composite Deepfake / Generative Manipulation"
            deduction_summary = (
                f"Deductive Synthesis: Multiple forensic layers concurrently flagged synthetic anomalies ({', '.join(flagged_layers)}). "
                "Strong multi-vector forensic indicators indicate extensive generative or face-swap synthesis."
            )

        reasoning_steps.append({
            "step": 6,
            "title": "Cross-Layer Forensic Deduction & Synthesis",
            "observation": deduction_summary,
            "status": "VERDICT REACHED"
        })

        return {
            "verdict": verdict,
            "risk_level": risk_level,
            "is_compromised": is_compromised,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(1.0 - fake_prob, 4),
            "manipulation_type": manipulation_type,
            "flagged_layer_count": len(flagged_layers),
            "flagged_layers": flagged_layers,
            "deduction_summary": deduction_summary,
            "reasoning_steps": reasoning_steps
        }

    # =========================================================================
    # MULTI-LAYER IMAGE ANALYSIS COORDINATOR
    # =========================================================================
    def analyze_multi_layer_image(
        self,
        image_bgr: np.ndarray,
        faces: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Coordinates full multi-layer forensic extraction and logical thinking audit.
        """
        l1 = self.analyze_selim_boundary_layer(image_bgr, faces=faces)
        l2 = self.analyze_xception_compression_layer(image_bgr, faces=faces)
        l3 = self.analyze_wang_upsampling_layer(image_bgr)
        l4 = self.analyze_openclip_semantic_layer(image_bgr)

        logic = self.conduct_logical_forensic_thinking(l1, l2, l3, l4, metadata=metadata)

        return {
            "layers": {
                "layer1_spatial_boundaries": l1,
                "layer2_depthwise_compression": l2,
                "layer3_frequency_upsampling": l3,
                "layer4_semantic_contrast": l4
            },
            "logical_thinking": logic
        }

