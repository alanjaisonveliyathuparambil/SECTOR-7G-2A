import os
import io
import base64
import cv2
import numpy as np
from typing import Dict, Any, List

from backend.face_detector import FaceDetector
from backend.forensics import ForensicEngine

class DetectionPipeline:
    def __init__(self, weights_dir: str = "weights"):
        self.detector = FaceDetector(scale_factor=1.35)
        self.engine = ForensicEngine(weights_dir=weights_dir)

    def _to_base64(self, image_bgr: np.ndarray, quality: int = 85) -> str:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', image_bgr, encode_param)
        if not success:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    def analyze_image_bytes(self, image_bytes: bytes, model_name: str = "ensemble") -> Dict[str, Any]:
        """
        Analyzes an uploaded image:
        - Evaluates global static image for GAN structural artifacts (Wang CNNDetection)
        - Detects all faces and evaluates facial deepfake authenticity (DFDC / Xception)
        - Returns GAN Synthetic Footprint Score and visual HUD overlays
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode uploaded image file.")

        h, w = img.shape[:2]
        
        # 1. Global GAN Structural Footprint Analysis (Sheng-Yu Wang CNNDetection)
        gan_footprint = self.engine.evaluate_gan_footprint(img)
        gan_score = gan_footprint["gan_synthetic_footprint_score"]

        # 2. OpenCLIP ViT-L-14 Latent Diffusion & Midjourney Detection
        diffusion_eval = self.engine.evaluate_diffusion_image(img)
        diffusion_score = diffusion_eval["diffusion_synthetic_probability"]

        # 3. Localized Facial Region Detection
        faces = self.detector.detect_faces(img)
        
        annotated = img.copy()
        face_results = []
        max_fake_prob = 0.0
        all_signals = []

        for idx, face_item in enumerate(faces):
            crop = face_item["crop"]
            res = self.engine.predict_crop(crop, model_name=model_name)
            
            fake_prob = res["fake_probability"]
            max_fake_prob = max(max_fake_prob, fake_prob)
            all_signals.append(res["signals"])
            
            bbox = face_item["bbox"]
            bx, by, bw, bh = bbox
            
            if fake_prob >= 0.70:
                color = (40, 40, 240)    # BGR Red
                label = f"FAKE #{idx+1}: {fake_prob*100:.1f}%"
            elif fake_prob >= 0.45:
                color = (0, 165, 255)    # BGR Amber
                label = f"SUSP #{idx+1}: {fake_prob*100:.1f}%"
            else:
                color = (230, 200, 20)   # BGR Cyan / Greenish
                label = f"REAL #{idx+1}: {res['real_probability']*100:.1f}%"

            # Draw futuristic HUD corners and bounding box
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)
            corner_len = max(int(min(bw, bh) * 0.15), 10)
            cv2.line(annotated, (bx, by), (bx + corner_len, by), color, 4)
            cv2.line(annotated, (bx, by), (bx, by + corner_len), color, 4)
            cv2.line(annotated, (bx + bw, by), (bx + bw - corner_len, by), color, 4)
            cv2.line(annotated, (bx + bw, by), (bx + bw, by + corner_len), color, 4)
            cv2.line(annotated, (bx, by + bh), (bx + corner_len, by + bh), color, 4)
            cv2.line(annotated, (bx, by + bh), (bx, by + bh - corner_len), color, 4)
            cv2.line(annotated, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), color, 4)
            cv2.line(annotated, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), color, 4)

            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (bx, max(0, by - 24)), (bx + tw + 10, by), color, -1)
            cv2.putText(annotated, label, (bx + 5, max(16, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            face_results.append({
                "face_index": idx + 1,
                "bbox": bbox,
                "fake_probability": fake_prob,
                "real_probability": res["real_probability"],
                "verdict": res["verdict"],
                "risk_level": res["risk_level"],
                "model_scores": res["model_scores"],
                "signals": res["signals"],
                "crop_preview": self._to_base64(crop, quality=80)
            })

        # Overall synthesis likelihood: combines localized face score, GAN footprint, and Diffusion detector
        if model_name == "wang_cnndetect":
            overall_fake = gan_score
        elif model_name == "clip_diffusion":
            overall_fake = diffusion_score
        elif len(faces) == 0:
            # For non-face or landscape/AI art images, fuse GAN and Diffusion scores
            overall_fake = 0.60 * diffusion_score + 0.40 * gan_score
        else:
            # Multi-modal consensus across faces, GAN footprints, and Diffusion signatures
            overall_fake = max(max_fake_prob, 0.55 * max_fake_prob + 0.25 * diffusion_score + 0.20 * gan_score)
            
        overall_fake = float(np.clip(overall_fake, 0.01, 0.99))

        if overall_fake >= 0.70:
            overall_verdict = "DEEPFAKE / AI-GENERATED"
            overall_risk = "CRITICAL"
        elif overall_fake >= 0.45:
            overall_verdict = "SUSPICIOUS MANIPULATION"
            overall_risk = "MODERATE"
        else:
            overall_verdict = "VERIFIED AUTHENTIC"
            overall_risk = "LOW"

        avg_signals = {}
        if all_signals:
            for k in all_signals[0].keys():
                avg_signals[k] = round(float(np.mean([s[k] for s in all_signals])), 3)
        else:
            avg_signals = {
                "spectral_anomaly": gan_footprint["spectral_lattice_anomaly"],
                "spatial_discontinuity": 0.15,
                "chrominance_mismatch": 0.12,
                "compression_artifact_index": 0.20
            }

        return {
            "media_type": "image",
            "resolution": f"{w}x{h}",
            "model_used": model_name,
            "faces_detected": len(faces),
            "fake_probability": round(overall_fake, 4),
            "real_probability": round(1.0 - overall_fake, 4),
            "gan_synthetic_footprint_score": gan_score,
            "gan_footprint": gan_footprint,
            "diffusion_detection": diffusion_eval,
            "verdict": overall_verdict,
            "risk_level": overall_risk,
            "faces": face_results,
            "aggregate_signals": avg_signals,
            "annotated_image": self._to_base64(annotated, quality=85)
        }

    def analyze_video_file(self, video_path: str, model_name: str = "ensemble", max_frames: int = 16) -> Dict[str, Any]:
        """
        Analyzes a video across frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)
        duration_sec = total_frames / fps

        if total_frames <= max_frames:
            sample_indices = list(range(max_frames))
        else:
            sample_indices = [int(i) for i in np.linspace(0, total_frames - 1, max_frames)]

        frame_timeline = []
        all_fake_scores = []
        keyframe_annotated = None
        max_seen_prob = 0.0

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            timestamp = round(frame_idx / fps, 2)
            faces = self.detector.detect_faces(frame)
            
            frame_max_fake = 0.0
            for face in faces:
                res = self.engine.predict_crop(face["crop"], model_name=model_name)
                frame_max_fake = max(frame_max_fake, res["fake_probability"])

            all_fake_scores.append(frame_max_fake)
            
            if frame_max_fake >= max_seen_prob or keyframe_annotated is None:
                max_seen_prob = frame_max_fake
                annotated = frame.copy()
                for face in faces:
                    bx, by, bw, bh = face["bbox"]
                    c = (40, 40, 240) if frame_max_fake >= 0.50 else (230, 200, 20)
                    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), c, 2)
                keyframe_annotated = self._to_base64(annotated, quality=75)

            frame_timeline.append({
                "timestamp_sec": timestamp,
                "frame_number": frame_idx,
                "fake_probability": round(frame_max_fake, 3),
                "faces_in_frame": len(faces)
            })

        cap.release()

        if not all_fake_scores:
            all_fake_scores = [0.1]

        mean_prob = float(np.mean(all_fake_scores))
        peak_prob = float(np.max(all_fake_scores))
        overall_fake = round(0.6 * peak_prob + 0.4 * mean_prob, 4)
        overall_real = round(1.0 - overall_fake, 4)

        if overall_fake >= 0.70:
            verdict = "DEEPFAKE DETECTED"
            risk_level = "CRITICAL"
        elif overall_fake >= 0.45:
            verdict = "SUSPICIOUS MANIPULATION"
            risk_level = "MODERATE"
        else:
            verdict = "VERIFIED AUTHENTIC"
            risk_level = "LOW"

        return {
            "media_type": "video",
            "duration_sec": round(duration_sec, 2),
            "total_frames": total_frames,
            "sampled_frames": len(frame_timeline),
            "model_used": model_name,
            "fake_probability": overall_fake,
            "real_probability": overall_real,
            "verdict": verdict,
            "risk_level": risk_level,
            "timeline": frame_timeline,
            "keyframe_image": keyframe_annotated or ""
        }
