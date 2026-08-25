"""Frame preprocessing: auto gamma, CLAHE contrast, light denoising."""

import cv2
import numpy as np


# Reused across frames: CLAHE object and one LUT per quantized gamma level
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_LUT_CACHE: dict[int, np.ndarray] = {}


def _gamma_lut(gamma: float) -> np.ndarray:
    """Return a cached LUT for gamma rounded to 2 decimals."""
    key = int(round(gamma * 100))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        g = key / 100.0
        lut = np.array(
            [pow(i / 255.0, g) * 255 for i in range(256)]
        ).astype("uint8")
        _LUT_CACHE[key] = lut
    return lut


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Brighten/dim, boost contrast, and lightly denoise one frame."""

    # Auto gamma correction — brighten dark frames, dim overexposed
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    brightness = l.mean()
    # gamma mapping: 0.6-1.4 range, 1.0 at mid brightness ~128
    gamma = 1.0 + (0.4 * (128.0 - brightness) / 128.0)
    gamma = float(np.clip(gamma, 0.6, 1.4))
    l = cv2.LUT(l, _gamma_lut(gamma))

    # CLAHE contrast enhancement on luminance
    l = _CLAHE.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Light denoising (bilateral filter preserves edges)
    return cv2.bilateralFilter(
        enhanced, d=5, sigmaColor=30, sigmaSpace=30
    )
