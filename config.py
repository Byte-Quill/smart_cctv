"""
========================================================================
 SMART CCTV — CENTRAL CONFIGURATION
========================================================================

Single source of truth for every tunable in the system.

All values are read at import time, so edit this file and restart
``main.py`` (or ``register.py``) for changes to take effect.

Sections at a glance:
    1. Runtime paths
    2. Camera
    3. Face detection
    4. Face recognition
    5. Face tracking
    6. Motion gate
    7. Scene analysis (YOLO)
    8. Alarm & timing
    9. Snapshots & logging
    10. Registration quality
"""

# ----------------------------------------------------------------------
# 0. PERFORMANCE PROFILE (low / balanced / high)
# ----------------------------------------------------------------------
# Pick the profile that matches your hardware. Every knob it touches is
# an EXISTING setting further down this file, so switching profiles never
# removes a feature — it just runs the same pipeline on lighter or
# heavier settings.
#
#   "low"      — small devices: ~4 GB RAM, 2-core CPU, Raspberry Pi,
#                old laptops. Smallest frames, cheapest detection, no
#                denoise, YOLO off. Smooth video is the priority.
#   "balanced" — typical laptops/desktops (4-8 cores). Default quality
#                with moderate frame-skipping.
#   "high"     — strong multi-core machines, ideally with a GPU. Full
#                resolution, detection every frame, CNN fallback and
#                YOLO allowed.
PERFORMANCE_MODE = "low"

# Fail loudly on a typo instead of silently running with default values.
if PERFORMANCE_MODE not in ("low", "balanced", "high"):
    raise ValueError(
        f"PERFORMANCE_MODE must be 'low', 'balanced', or 'high', "
        f"got {PERFORMANCE_MODE!r}"
    )

# Back-compat flag: True when running the lightest profile.
LOW_POWER = PERFORMANCE_MODE == "low"

# ----------------------------------------------------------------------
# 0b. HARDWARE PROFILE (future device portability)
# ----------------------------------------------------------------------
# The system is designed to move to smaller boards later (Raspberry Pi 5,
# or an ESP32-CAM running a companion capture service). All device-
# specific behaviour (camera backend, siren output, GPIO relay) lives
# behind cctv/hardware.py, which reads this setting.
#
#   "pc"     — desktop/laptop: OpenCV camera + pygame WAV siren (default)
#   "pi"     — Raspberry Pi 5: same stack, plus optional GPIO relay hook
#   "esp32"  — ESP32-CAM: frames arrive over a network stream; the heavy
#              pipeline still runs on a host machine
HARDWARE_PROFILE = "pc"

# ----------------------------------------------------------------------
# 1. RUNTIME PATHS
# ----------------------------------------------------------------------
# One sub-folder per registered family member, each holding that person's
# face photos (created by register.py) used to build the recognition DB.
FAMILY_DIR = "family"

# Where snapshot JPGs of unknown people are saved.
SNAPSHOT_DIR = "snapshots"

# Where the SQLite event DB (events.db) and text audit log (security.log) live.
LOG_DIR = "logs"

# Delete snapshots, database rows, and log lines older than this many days.
RETENTION_DAYS = 30

# ----------------------------------------------------------------------
# 2. CAMERA
# ----------------------------------------------------------------------
# Which camera to use (0 = built-in/webcam, 1+ for extra devices).
CAMERA_INDEX = 0

# Mirror the video horizontally so the view behaves like a mirror:
# raising your right hand appears on the RIGHT side of the screen.
MIRROR_DISPLAY = True

# Capture resolution requested from the camera.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ----------------------------------------------------------------------
# 2b. FRAME ENHANCEMENT
# ----------------------------------------------------------------------
# Apply the bilateral denoise step in enhance_frame(). It improves
# recognition slightly but is the most expensive per-frame CPU step,
# so the low profile turns it off.
ENHANCE_DENOISE = True

# Show a live FPS counter in the top-right corner of the window.
SHOW_FPS = True

# ----------------------------------------------------------------------
# 3. FACE DETECTION
# ----------------------------------------------------------------------
# Face detection runs on frames downscaled by this factor for speed.
# 0.5 = detect on a half-size frame (faster but slightly less accurate).
DETECTION_SCALE = 0.5

# Minimum face height in pixels (at detection resolution) to be valid.
# Filters out tiny false-positive detections.
MIN_FACE_SIZE = 40

# If HOG finds no faces, fall back to the slower CNN model on the
# full-resolution frame.
ENABLE_CNN_FALLBACK = True

# ----------------------------------------------------------------------
# 4. FACE RECOGNITION
# ----------------------------------------------------------------------
# Maximum face_distance for a match to count. Lower = stricter.
# dlib distances are roughly 0.0-0.6 for the same person.
FACE_TOLERANCE = 0.45

# ----------------------------------------------------------------------
# 5. FACE TRACKING
# ----------------------------------------------------------------------
# Run full face detection every N frames (1 = every frame); tracks
# persist in between via centroid matching + patience.
TRACKING_SKIP_FRAMES = 2

# How many frames to keep a track alive after the face disappears.
TRACKING_PATIENCE = 5

# EMA alpha for bounding-box smoothing (0.0-1.0; higher = box follows the
# detection faster but is jumpier).
TRACKING_SMOOTH_ALPHA = 0.6

# Temporal ensemble: classify a tracked face by majority vote over the
# last N frames, smoothing out single-frame mis-detections.
ENSEMBLE_FRAMES = 5

# ----------------------------------------------------------------------
# 6. MOTION GATE
# ----------------------------------------------------------------------
# When enabled, skip the expensive pipeline unless the scene changes.
MOTION_ENABLED = True

# Pixel-difference threshold (0-255) for a pixel to count as "moved".
MOTION_THRESHOLD = 25.0

# Minimum fraction of changed pixels needed to count as motion (0-1).
MOTION_MIN_AREA = 0.01

# Resolution scale used for the cheap motion diff check (small = faster).
MOTION_SCALE = 0.25

# How fast the running background model adapts to the scene (0-1).
# A lower value makes the gate more sensitive to slow, gradual movement.
MOTION_BG_ALPHA = 0.05

# ----------------------------------------------------------------------
# 7. SCENE ANALYSIS (YOLO)
# ----------------------------------------------------------------------
# Run YOLOv8 to explain the scene while an unknown face lingers.
# Detected animals suppress the siren; a confirmed human shortens the delay.
ANIMAL_DETECTION_ENABLED = True

# Re-run YOLO only every N frames while an unknown face is present.
YOLO_SKIP_FRAMES = 3

# ----------------------------------------------------------------------
# 8. ALARM & TIMING
# ----------------------------------------------------------------------
# Consecutive frames with an unknown face before it is "confirmed".
UNKNOWN_CONFIRMATIONS = 5

# How long (seconds) a confirmed unknown must linger before the siren.
UNKNOWN_DELAY_SECONDS = 10

# Faster delay used when YOLO confirms a human is present.
UNKNOWN_HUMAN_DELAY_SECONDS = 1

# Faster delay used in night security mode.
NIGHT_UNKNOWN_DELAY_SECONDS = 2

# Alarm sound file (relative to the project root).
SIREN_FILE = "sounds/siren.wav"

# ----------------------------------------------------------------------
# 8b. SIREN AUTO-SHUTDOWN & NEPAL-TIME SECURITY MODES
# ----------------------------------------------------------------------
# The siren never runs forever. Once triggered it auto-stops after a
# fixed duration, and a family member can silence it early (press "s" in
# the window, or call Siren.stop()).
#
# Durations depend on the time of day, using NEPAL TIME (NPT, UTC+5:45):
#   • Daytime   (06:00-22:00 NPT) → siren runs SIREN_DAY_DURATION.
#   • Night mode (22:00-06:00 NPT) → SECURITY MODE arms automatically at
#     NIGHT_START_HOUR; the siren runs the longer SIREN_NIGHT_DURATION.
#
# All clock checks go through cctv/timeutil.py so the whole system uses
# one consistent Nepal-time clock regardless of the host's timezone.
# Siren auto-stop duration during the day (seconds). 2 minutes.
SIREN_DAY_DURATION = 120

# Siren auto-stop duration in night security mode (seconds). 5 minutes.
SIREN_NIGHT_DURATION = 300

# After the siren auto-stops, wait this long before it may re-trigger,
# even if the unknown person is still present. Prevents an endless
# stop/restart cycle and gives the family time to respond.
SIREN_RETRIGGER_COOLDOWN = 60

# Hour (Nepal time) when night security mode begins. 22 = 10 PM.
NIGHT_START_HOUR = 22

# Hour (Nepal time) when night security mode ends and day mode resumes.
NIGHT_END_HOUR = 6

# Nepal Time is a fixed UTC+5:45 offset (no daylight saving).
NEPAL_UTC_OFFSET_MINUTES = 345

# ----------------------------------------------------------------------
# 9. SNAPSHOTS & LOGGING
# ----------------------------------------------------------------------
# Save an unknown snapshot every this many seconds.
SNAPSHOT_INTERVAL = 5

# Minimum seconds between two FAMILY_SIGHTING log entries per person.
SIGHTING_LOG_INTERVAL = 30

# ----------------------------------------------------------------------
# 10. REGISTRATION QUALITY (register.py)
# ----------------------------------------------------------------------
# Number of photos to auto-capture per person.
TARGET_PHOTOS = 10

# Consecutive good frames needed before a photo is snapped automatically.
AUTO_CAPTURE_STABLE_FRAMES = 8

# Minimum face height (full resolution) for a registration capture.
MIN_REG_FACE_SIZE = 80

# Laplacian variance floor — frames below this are rejected as blurry.
BLUR_THRESHOLD = 80

# Mean pixel brightness range (0-255) accepted for a capture.
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 215

# Minimum face_distance from every known capture before a new pose is
# kept, rejecting near-duplicate poses during a registration session.
MIN_ENCODING_DISTANCE = 0.25

# ----------------------------------------------------------------------
# PERFORMANCE PROFILE OVERRIDES
# ----------------------------------------------------------------------
# Applied at import time AFTER every value above is set, so object names
# used throughout the code keep working unchanged. Set PERFORMANCE_MODE
# at the top of this file to "low", "balanced", or "high". Nothing here
# adds, removes, or changes any code path — it only picks values for the
# knobs the pipeline already reads.
if PERFORMANCE_MODE == "low":

    # 2. Camera — fewer pixels per frame means less RAM and CPU.
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # 2b. Enhancement — skip the expensive bilateral denoise; gamma and
    # CLAHE are cheap LUT/lookup ops and stay on.
    ENHANCE_DENOISE = False

    # 3. Face detection — HOG on a smaller image is much cheaper.
    DETECTION_SCALE = 0.4
    MIN_FACE_SIZE = 32
    # HOG is fast; the CNN model needs the big dlib model plus GPU-like
    # CPU effort and is the main reason to skip it on a small box.
    ENABLE_CNN_FALLBACK = False

    # 5. Face tracking — run full detection less often; tracks bridge
    # the gap, so every individual frame no longer pays HOG's cost.
    TRACKING_SKIP_FRAMES = 4

    # 6. Motion gate — smaller diff image, cheaper blur, and a slightly
    # higher threshold so the heavy pipeline is skipped more aggressively.
    MOTION_SCALE = 0.2
    MOTION_BG_ALPHA = 0.08
    MOTION_THRESHOLD = 32.0

    # 7. Scene analysis (YOLO) — disable the huge model. The ultralytics
    # dependency and its detector object can then be omitted entirely,
    # letting a minimal install run without the large weights file.
    ANIMAL_DETECTION_ENABLED = False

elif PERFORMANCE_MODE == "balanced":

    # Default-quality values with moderate frame-skipping.
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    ENHANCE_DENOISE = True
    DETECTION_SCALE = 0.5
    MIN_FACE_SIZE = 40
    ENABLE_CNN_FALLBACK = False
    TRACKING_SKIP_FRAMES = 2
    MOTION_SCALE = 0.25
    MOTION_BG_ALPHA = 0.05
    MOTION_THRESHOLD = 25.0
    ANIMAL_DETECTION_ENABLED = True

elif PERFORMANCE_MODE == "high":

    # Full quality: every frame processed, CNN fallback and YOLO allowed.
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    ENHANCE_DENOISE = True
    DETECTION_SCALE = 0.5
    MIN_FACE_SIZE = 40
    ENABLE_CNN_FALLBACK = True
    TRACKING_SKIP_FRAMES = 1
    MOTION_SCALE = 0.25
    MOTION_BG_ALPHA = 0.05
    MOTION_THRESHOLD = 25.0
    ANIMAL_DETECTION_ENABLED = True