# Which camera to use (0 = built-in/webcam)
CAMERA_INDEX = 0

# Max distance for a face to count as a match
FACE_TOLERANCE = 0.45

# Frames needed before treating someone as unknown
UNKNOWN_CONFIRMATIONS = 5

# Minimum seconds between two FAMILY_SIGHTING log entries per person
SIGHTING_LOG_INTERVAL = 30

# How long an unknown person must stay before alarm
UNKNOWN_DELAY_SECONDS = 10

# Faster night alarm — how many seconds before siren triggers outside allowed hours
NIGHT_UNKNOWN_DELAY_SECONDS = 2

# Hours when the system is allowed to be used
ALLOWED_START_HOUR = 6
ALLOWED_END_HOUR = 22

# Save an unknown snapshot every X seconds
SNAPSHOT_INTERVAL = 5

# Face detection resolution scale (0.5 = detect on half-size frame)
DETECTION_SCALE = 0.5

# Minimum face height in pixels (at detection resolution) to consider valid
MIN_FACE_SIZE = 40

# Registration quality thresholds
MIN_REG_FACE_SIZE = 80       # min face height (full-res)
BLUR_THRESHOLD = 80           # Laplacian variance floor
MIN_BRIGHTNESS = 40           # mean pixel brightness 0-255
MAX_BRIGHTNESS = 215
MIN_ENCODING_DISTANCE = 0.25  # min distance to reject duplicate pose
AUTO_CAPTURE_STABLE_FRAMES = 8

# Enable CNN fallback detection when HOG finds nothing
ENABLE_CNN_FALLBACK = True

# Temporal ensemble: classify based on majority vote over N frames per tracked face
ENSEMBLE_FRAMES = 5

# Face tracking: run full detection every N frames (1 = every frame)
TRACKING_SKIP_FRAMES = 2

# Motion gate: skip the heavy pipeline when nothing moves
MOTION_ENABLED = True
MOTION_THRESHOLD = 25.0   # pixel-diff threshold (0-255)
MOTION_MIN_AREA = 0.01    # min fraction of changed pixels to count as motion
MOTION_SCALE = 0.25       # resolution used for the cheap diff check
MOTION_BG_ALPHA = 0.05    # how fast the background model adapts (0-1)

# YOLO object detection: run every N frames while an unknown face lingers
YOLO_SKIP_FRAMES = 3

# Exponential moving average alpha for bounding box smoothing (0.0-1.0, higher = faster tracking)
TRACKING_SMOOTH_ALPHA = 0.6

# How many frames to keep a track alive after the face disappears
TRACKING_PATIENCE = 5

# Folder paths
FAMILY_DIR = "family"
SNAPSHOT_DIR = "snapshots"
LOG_DIR = "logs"

# Retention: delete snapshots/log entries older than this many days
RETENTION_DAYS = 30

# Alarm sound file
SIREN_FILE = "sounds/siren.wav"

# New parameters for animal detection and human delay
ANIMAL_DETECTION_ENABLED = True
UNKNOWN_HUMAN_DELAY_SECONDS = 1