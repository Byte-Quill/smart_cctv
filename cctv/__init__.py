"""
=====================================================================
 cctv — Smart CCTV processing pipeline
=====================================================================

This package contains every building block of the Smart CCTV system.
The top-level scripts (``main.py``, ``register.py``) orchestrate these
modules; nothing here should talk to a camera by itself.

PIPELINE (per camera frame, in ``main.py``)

    camera → enhance → detect → recognize → track → alarm/log
                │          │                       │
                │          └── faces.py            └── tracking.py
                └── enhance.py
    plus:  motion.py   (cheap gate BEFORE enhance)
           yolo.py     (scene analysis while an unknown lingers)
           siren.py    (loud alarm output)
           storage.py  (SQLite + file event persistence)
           quality.py  (guards used by register.py)
           hud.py      (draws the on-screen overlay in main.py)

MODULE RESPONSIBILITIES
    enhance   — brightness/contrast/denoise frame preprocessing.
    faces     — load the family DB, detect (HOG + CNN fallback), recognize.
    tracking  — temporal smoothing of boxes + majority-vote identity.
    motion    — frame-difference gate that skips the heavy pipeline.
    yolo      — YOLOv8 animal/human classification (false-alarm suppression).
    siren     — looping WAV alarm with thread-safe on/off.
    storage   — SQLite events table + text audit log + retention.
    quality   — blur/brightness/size/duplicate checks for registration.
    hud       — all cv2 overlay drawing used by the main loop.

Public classes/functions are re-exported below for convenience.
"""

# Public API re-export — allows ``from cctv import MotionDetector`` etc.
from .motion import MotionDetector
from .tracking import FaceTrack, FaceHistory, match_tracks
from .faces import load_family_database, recognize_face, detect_faces_enhanced
from .enhance import enhance_frame
from .siren import Siren
from .yolo import ObjectDetector
from .storage import (
    initialize_database,
    log_event,
    enforce_retention,
    logger,
)
from .quality import (
    estimate_blur,
    brightness_ok,
    blur_ok,
    face_large_enough,
    compute_encoding,
    load_existing_encodings,
    is_duplicate_pose,
)
from .hud import (
    draw_face_boxes,
    draw_countdown,
    draw_family_text,
    draw_mode,
    draw_status,
)

__all__ = [
    # motion
    "MotionDetector",
    # tracking
    "FaceTrack",
    "FaceHistory",
    "match_tracks",
    # faces
    "load_family_database",
    "recognize_face",
    "detect_faces_enhanced",
    # enhance
    "enhance_frame",
    # siren
    "Siren",
    # yolo
    "ObjectDetector",
    # storage
    "initialize_database",
    "log_event",
    "enforce_retention",
    "logger",
    # quality
    "estimate_blur",
    "brightness_ok",
    "blur_ok",
    "face_large_enough",
    "compute_encoding",
    "load_existing_encodings",
    "is_duplicate_pose",
    # hud
    "draw_face_boxes",
    "draw_countdown",
    "draw_family_text",
    "draw_mode",
    "draw_status",
]
