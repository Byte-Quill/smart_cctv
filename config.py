# Which camera to use (0 = built-in/webcam)
CAMERA_INDEX = 0

# Max distance for a face to count as a match
FACE_TOLERANCE = 0.45

# Frames needed before accepting a family member
REQUIRED_CONFIRMATIONS = 5

# Frames needed before treating someone as unknown
UNKNOWN_CONFIRMATIONS = 5

# How long an unknown person must stay before alarm
UNKNOWN_DELAY_SECONDS = 10

# Hours when the system is allowed to be used
ALLOWED_START_HOUR = 6
ALLOWED_END_HOUR = 22

# Save an unknown snapshot every X seconds
SNAPSHOT_INTERVAL = 5

# Scale factor to shrink frames for faster recognition
FRAME_SCALE = 0.25

# Folder paths
FAMILY_DIR = "family"
SNAPSHOT_DIR = "snapshots"
LOG_DIR = "logs"

# Alarm sound file
SIREN_FILE = "sounds/siren.wav"