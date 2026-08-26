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
         siren.py  → loud looping WAV alarm (thread-safe, auto-stops)
         storage.py→ SQLite event DB + text audit log + retention
         hud.py    → draws every on-screen overlay used by main.py

    Cross-cutting services:

         timeutil.py → Nepal Time (UTC+5:45) clock; day vs night security
                       mode and siren durations all read from here
         hardware.py → device abstraction (pc / pi / esp32) so the system
                       can move to a Raspberry Pi 5 or ESP32-CAM later

    Registration quality gates (used by ``register.py`` and the in-app
    enrollment flow in ``enroll.py``):

         quality.py → blur / brightness / size / duplicate-pose checks
         enroll.py  → in-app 'Add Family' flow: on-screen name entry +
                      guided left→center→right zone capture, opened from
                      the main window's button (or the 'a' key)

MODULE → JOB
    enhance   — brightness/contrast/denoise frame preprocessing.
    faces     — load the family DB, detect (HOG + CNN fallback), recognize.
    tracking  — temporal smoothing of boxes + majority-vote identity.
    motion    — background-model motion gate that skips the heavy pipeline.
    yolo      — YOLOv8 animal/human classification (false-alarm suppression).
    siren     — looping WAV alarm with thread-safe on/off + auto-stop timer.
    storage   — SQLite events table + text audit log + retention.
    quality   — blur/brightness/size/duplicate checks for registration.
    enroll    — in-app add-family flow (name entry + guided zone capture).
    hud       — all cv2 overlay drawing used by the main loop.
    timeutil  — Nepal-time clock, day/night security mode, siren durations.
    hardware  — camera/device abstraction for future Pi/ESP32 ports.

Errors are handled defensively: a missing siren file, CNN model, or YOLO
weights degrades functionality gracefully instead of crashing the camera
loop.
"""
