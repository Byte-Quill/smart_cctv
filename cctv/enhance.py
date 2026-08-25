"""Frame preprocessing: auto gamma, CLAHE contrast, light denoising."""

import cv2
import numpy as np


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Brighten/dim, boost contrast, and lightly denoise one frame."""

    # Auto gamma correction — brighten dark frames, dim overexposed
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    brightness = l.mean()
    # gamma mapping: 0.6-1.4 range, 1.0 at mid brightness ~128
    gamma = 1.0 + (0.4 * (128.0 - brightness) / 128.0)
    gamma = np.clip(gamma, 0.6, 1.4)
    lut = np.array(
        [pow(i / 255.0, gamma) * 255 for i in range(256)]
    ).astype("uint8")
    l = cv2.LUT(l, lut)

    # CLAHE contrast enhancement on luminance
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Light denoising (bilateral filter preserves edges)
    return cv2.bilateralFilter(
        enhanced, d=5, sigmaColor=30, sigmaSpace=30
    )
