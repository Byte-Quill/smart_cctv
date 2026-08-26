"""Image-quality checks shared by the registration flow."""

import os

import cv2
import face_recognition
import numpy as np

from config import (
    MIN_REG_FACE_SIZE,
    BLUR_THRESHOLD,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_ENCODING_DISTANCE,
)


def estimate_blur(gray: np.ndarray) -> float:
    """Laplacian variance — lower values mean more blur."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_ok(brightness: float) -> bool:
    """True when mean brightness is inside the accepted range."""
    return MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS


def blur_ok(blur_value: float) -> bool:
    """True when the face region is sharp enough."""
    return blur_value >= BLUR_THRESHOLD


def face_large_enough(face_height: int) -> bool:
    """True when the face is at least the minimum registration size."""
    return face_height >= MIN_REG_FACE_SIZE


def compute_encoding(rgb_face: np.ndarray):
    """Return 128-d encoding or None if no face is visible.

    A tight face crop can cause dlib's landmark detector to produce
    points outside the image bounds, crashing compute_face_descriptor.
    We pad the crop with replicated border pixels to give dlib room.
    """
    h, w = rgb_face.shape[:2]
    if h < 10 or w < 10:
        return None

    # Pad 25% on each side to prevent landmark overflow on tight crops
    pad_y = max(h // 4, 10)
    pad_x = max(w // 4, 10)
    padded = cv2.copyMakeBorder(
        rgb_face, pad_y, pad_y, pad_x, pad_x,
        cv2.BORDER_REPLICATE
    )

    try:
        locs = face_recognition.face_locations(padded, model="hog")
        if len(locs) != 1:
            return None
        encs = face_recognition.face_encodings(padded, locs)
        return encs[0] if encs else None
    except Exception:
        return None


def load_existing_encodings(folder: str) -> list:
    """Load all previously captured photos in *folder* and return encodings."""
    encodings = []
    if not os.path.isdir(folder):
        return encodings
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = face_recognition.load_image_file(
            os.path.join(folder, fname)
        )
        enc = compute_encoding(img)
        if enc is not None:
            encodings.append(enc)
    return encodings


def is_duplicate_pose(
    encoding,
    known_encodings,
    threshold: float = MIN_ENCODING_DISTANCE
) -> bool:
    """True when the pose is too similar to an already-captured photo."""
    if encoding is None or not known_encodings:
        return False
    distances = face_recognition.face_distance(known_encodings, encoding)
    return float(distances.min()) < threshold
