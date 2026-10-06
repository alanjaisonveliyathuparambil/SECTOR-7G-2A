import os
import cv2
import numpy as np

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

def generate_face_image(is_fake: bool = False, filename: str = "sample.jpg"):
    # Create a 512x512 portrait image with realistic features for testing face detectors
    img = np.ones((512, 512, 3), dtype=np.uint8) * 45  # Studio dark backdrop
    
    # Backdrop vignette
    cv2.circle(img, (256, 256), 260, (70, 60, 55), -1)
    img = cv2.GaussianBlur(img, (101, 101), 30)

    # Head oval (skin tone)
    center = (256, 240)
    axes = (110, 150)
    skin_color = (180, 205, 240) if not is_fake else (160, 190, 235)  # BGR
    cv2.ellipse(img, center, axes, 0, 0, 360, skin_color, -1)
    
    # Hair
    cv2.ellipse(img, (256, 170), (125, 100), 0, 160, 380, (25, 20, 20), -1)
    cv2.ellipse(img, (140, 240), (25, 90), 0, 0, 360, (25, 20, 20), -1)
    cv2.ellipse(img, (372, 240), (25, 90), 0, 0, 360, (25, 20, 20), -1)

    # Eyes
    # Left eye
    cv2.ellipse(img, (205, 220), (20, 10), 0, 0, 360, (250, 250, 250), -1)
    cv2.circle(img, (205, 220), 8, (60, 40, 20), -1)
    cv2.circle(img, (203, 218), 3, (255, 255, 255), -1)
    # Right eye
    cv2.ellipse(img, (307, 220), (20, 10), 0, 0, 360, (250, 250, 250), -1)
    cv2.circle(img, (307, 220), 8, (60, 40, 20), -1)
    cv2.circle(img, (305, 218), 3, (255, 255, 255), -1)

    # Eyebrows
    cv2.ellipse(img, (205, 200), (26, 6), -5, 0, 180, (30, 20, 20), 4)
    cv2.ellipse(img, (307, 200), (26, 6), 5, 0, 180, (30, 20, 20), 4)

    # Nose
    cv2.line(img, (256, 225), (254, 265), (140, 165, 200), 3)
    cv2.ellipse(img, (256, 268), (14, 6), 0, 0, 180, (120, 145, 180), 2)

    # Mouth
    cv2.ellipse(img, (256, 315), (32, 14), 0, 0, 360, (110, 120, 200), -1)
    cv2.line(img, (226, 315), (286, 315), (70, 70, 140), 2)

    # Neck
    cv2.rectangle(img, (216, 370), (296, 480), (160, 185, 220), -1)
    # Shirt collar
    cv2.ellipse(img, (256, 510), (140, 90), 0, 0, 360, (40, 35, 30), -1)

    if is_fake:
        # Simulate common Deepfake manipulation artifacts:
        # 1. Subtle warping seam and color shift in the face mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(mask, center, (90, 120), 0, 0, 360, 255, -1)
        
        # Color temperature dissonance on swapped mask
        swapped = img.copy()
        swapped[:, :, 0] = np.clip(swapped[:, :, 0].astype(int) + 25, 0, 255) # Blue shift
        swapped[:, :, 2] = np.clip(swapped[:, :, 2].astype(int) - 20, 0, 255) # Red shift
        
        # Artificial blurring on boundaries (feathered blend)
        blurred_mask = cv2.GaussianBlur(mask, (35, 35), 12)
        alpha = (blurred_mask / 255.0)[:, :, np.newaxis]
        img = (swapped * alpha + img * (1 - alpha)).astype(np.uint8)
        
        # High frequency GAN checkerboard noise pattern
        grid = np.zeros((512, 512, 3), dtype=np.int16)
        grid[::4, ::4, :] = 35
        grid[1::4, 1::4, :] = -35
        img = np.clip(img.astype(np.int16) + (grid * alpha).astype(np.int16), 0, 255).astype(np.uint8)

    save_path = os.path.join(SAMPLES_DIR, filename)
    cv2.imwrite(save_path, img)
    print(f"Generated sample: {save_path} (is_fake={is_fake})")

if __name__ == "__main__":
    generate_face_image(is_fake=False, filename="authentic_portrait.jpg")
    generate_face_image(is_fake=True, filename="deepfake_synthetic_face.jpg")
