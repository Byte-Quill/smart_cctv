"""
=====================================================================
 cctv — Smart CCTV processing pipeline
=====================================================================

Every building block of the Smart CCTV system lives in this package.
The top-level scripts (``main.py``, ``register.py``) are thin, readable
orchestrators; all real work happens here.

There is exactly one responsibility per module, so you can understand
the whole system by reading this list once:

PIPELINE (per camera frame, in ``main.py``)

    camera → [motion gate] → enhance → detect → recognize → track
                 │              │          │               │
                 │          enhance.py    faces.py        tracking.py
                 │
             motion.py  (cheap skip-ahead: no motion → no heavy work)

    While an UNKNOWN person lingers, the system additionally consults:

         yolo.py   → animal vs human scene analysis (false-alarm guard)
         siren.py  → loud looping WAV alarm (thread-safe)
         storage.py→ SQLite event DB + text audit log + retention
         hud.py    → draws every on-screen overlay used by main.py

    Registration quality gates (used by ``register.py``):

         quality.py → blur / brightness / size / duplicate-pose checks

MODULE → JOB
    enhance   — brightness/contrast/denoise frame preprocessing.
    faces     — load the family DB, detect (HOG + CNN fallback), recognize.
    tracking  — temporal smoothing of boxes + majority-vote identity.
    motion    — background-model motion gate that skips the heavy pipeline.
    yolo      — YOLOv8 animal/human classification (false-alarm suppression).
    siren     — looping WAV alarm with thread-safe on/off.
    storage   — SQLite events table + text audit log + retention.
    quality   — blur/brightness/size/duplicate checks for registration.
    hud       — all cv2 overlay drawing used by the main loop.

Errors are handled defensively: a missing siren file, CNN model, or YOLO
weights degrades functionality gracefully instead of crashing the camera
loop.
"""
